"""
Fourier Neural Operator implementation for Geometric FNO.
"""

import torch
import torch.nn.functional as F
from loader import MeshToGridDataset
from torch import cfloat, einsum, fft, nn, optim, rand
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import os

torch.set_float32_matmul_precision("high")


def collate_fn(batch):
    """
    Static function to handle batches of meshes with variable number of nodes.
    To be used in DataLoader: collate_fn=collate_fn
    """
    # 1. Find max number of nodes in this batch
    max_n = max(item["y_mesh"].shape[0] for item in batch)

    x_grids = []
    y_meshes = []
    coords = []
    masks = []

    for item in batch:
        # x_grid is fixed (C, H, W)
        x_grids.append(item["x_grid"])

        # Retrieve variable data
        y = item["y_mesh"]  # (N, out_channels)
        c = item["query_coords"]  # (N, 2)
        n = y.shape[0]

        padding_len = max_n - n

        # Mask creation: 1 = Real data, 0 = Padding
        mask = torch.cat([torch.ones(n), torch.zeros(padding_len)], dim=0)
        masks.append(mask)

        # Data Padding
        if padding_len > 0:
            # Zero padding at the end
            y_pad = torch.cat([y, torch.zeros(padding_len, y.shape[1])], dim=0)
            c_pad = torch.cat([c, torch.zeros(padding_len, c.shape[1])], dim=0)
        else:
            y_pad = y
            c_pad = c

        y_meshes.append(y_pad)
        coords.append(c_pad)

    return {
        "x_grid": torch.stack(x_grids),  # (B, C, H, W)
        "y_mesh": torch.stack(y_meshes),  # (B, Max_N, out_channels)
        "query_coords": torch.stack(coords),  # (B, Max_N, 2)
        "mask": torch.stack(masks),  # (B, Max_N) -> NUOVA CHIAVE
    }


class SpectralConv2d(nn.Module):
    """
    Two dimensional spectral convolution.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * rand(in_channels, out_channels, self.modes1, self.modes2, dtype=cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale
            * rand(in_channels, out_channels, self.modes1, self.modes2, dtype=cfloat)
        )

    def compl_mul2d(self, x_in, weights):
        """
        Complex multiplication between input and weights.
        """
        return einsum("bixy,ioxy->boxy", x_in, weights)

    def forward(self, x):
        """
        Forward pass of SpectralConv2d.
        """
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of 2.
        x_ft = fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Upper corners
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        # Lower corners
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class GeoFNO(nn.Module):
    """
    Geometric Fourier Neural Operator model.
    """

    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Input: k, Q, x, y, mask (5 channels)
        self.fc0 = nn.Conv2d(5, self.width, 1)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 4)  # Output: T, u, v, P

    def forward(self, x_grid, query_coords):
        """
        Forward pass of GeoFNO.
        """
        # 1. Processing on Latent Grid
        # x_grid shape: (Batch, 5, H, W)
        x = self.fc0(x_grid)  # (B, width, H, W)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2  # Latent output on grid (B, width, H, W)

        # 2. Geometric Querying (Grid -> Mesh)
        # query_coords shape: (B, N_points, 2), grid_sample expects (B, 1, N_points, 2)
        q = query_coords.unsqueeze(1)

        # Interpolate grid values at mesh node locations
        x_sampled = F.grid_sample(
            x, q, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        # x_sampled shape: (B, width, 1, N_points)

        x_sampled = x_sampled.squeeze(2).permute(0, 2, 1)  # (B, N_points, width)

        # 3. Final Decoder (point-wise)
        x_sampled = self.fc1(x_sampled)
        x_sampled = F.gelu(x_sampled)
        out = self.fc2(x_sampled)  # (B, N_points, 4)

        return out


import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import numpy as np


# Assuming GeoFNO and MeshToGridDataset are imported or defined in the same file
# from loader import MeshToGridDataset
# from your_script import GeoFNO, collate_fn

def compute_masked_loss(pred, target, mask):
    """
    Computes MSE Loss ignoring padding.
    pred, target: (B, N, C)
    mask: (B, N) -> 1 for real data, 0 for padding
    """
    # Expand mask to cover channels (C)
    # mask becomes (B, N, 1) for broadcasting over (B, N, C)
    mask_expanded = mask.unsqueeze(-1)

    # Compute element-wise squared error
    squared_error = (pred - target) ** 2

    # Zero out error in padding zones
    masked_squared_error = squared_error * mask_expanded

    # Sum of errors divided by number of valid elements (sum of mask * channels)
    loss = masked_squared_error.sum() / (mask.sum() * target.shape[-1])
    return loss


def compute_relative_l2(pred, target, mask):
    """
    Computes Relative L2 Error (Standard metric for FNO).
    Lower is better. Acts as an accuracy proxy.
    """
    mask_expanded = mask.unsqueeze(-1)

    # Apply mask
    pred = pred * mask_expanded
    target = target * mask_expanded

    # L2 Norm calculation over each sample in batch
    # Sum over nodes (dim 1) and channels (dim 2), keep batch (dim 0)
    diff_norms = torch.norm(pred - target, p=2, dim=(1, 2))
    target_norms = torch.norm(target, p=2, dim=(1, 2))

    # Avoid division by zero
    rel_l2 = diff_norms / (target_norms + 1e-8)

    return rel_l2.mean()  # Mean over batch


def plot_history(train_losses, test_errors, save_dir):
    """
    Saves a plot of training loss and test error history.
    """
    plt.figure(figsize=(10, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy (Rel L2)
    plt.subplot(1, 2, 2)
    plt.plot(test_errors, color='orange', label='Test Error (Rel L2)')
    plt.xlabel('Epoch')
    plt.ylabel('Rel L2 Error')
    plt.title('Test Error (Lower is better)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"))
    plt.close()


# Aggiunto parametro domain_bounds
def visualize_sample(coords, target, pred, mask, epoch, save_dir, len_x=20, len_y=5):
    """
    Genera plot in stile FEM, denormalizzando le coordinate per mostrare
    l'aspect ratio fisico corretto.
    """

    # 2. Preparazione Dati (primo elemento del batch)
    # c_norm sono le coordinate normalizzate nel range latente [-1, 1]
    c_norm = coords[0].cpu().numpy()  # (N_max, 2)
    t = target[0].cpu().numpy()  # (N_max, 4)
    p = pred[0].cpu().numpy()  # (N_max, 4)
    m = mask[0].cpu().numpy()  # (N_max)

    # Filtra via il padding
    valid_indices = m == 1
    c_valid = c_norm[valid_indices]
    t_vals = t[valid_indices]
    p_vals = p[valid_indices]

    # --- INIZIO DENORMALIZZAZIONE GEOMETRICA ---

    # Passo A: Dal range latente [-1, 1] al range normalizzato [0, 1]
    # Formula inversa di: query = norm * 2 - 1
    c_01 = (c_valid + 1) / 2.0

    # Passo B: Dal range [0, 1] alle Unità Fisiche
    # Formula inversa di: norm_x = (phys_x - x_min) / len_x
    phys_x = c_01[:, 0] * len_x
    phys_y = c_01[:, 1] * len_y

    # --- FINE DENORMALIZZAZIONE GEOMETRICA ---

    # 3. Ricostruzione Mesh (Triangolazione sulle coordinate FISICHE)
    # Ora la triangolazione avverrà sulla geometria "distesa" correttamente
    triang = tri.Triangulation(phys_x, phys_y)

    # 4. Estrazione Campi Fisici (come prima...)
    T_gt = t_vals[:, 0]
    vel_mag_gt = np.sqrt(t_vals[:, 2] ** 2 + t_vals[:, 3] ** 2)
    T_pred = p_vals[:, 0]
    vel_mag_pred = np.sqrt(p_vals[:, 2] ** 2 + p_vals[:, 3] ** 2)
    T_err = np.abs(T_gt - T_pred)
    vel_err = np.abs(vel_mag_gt - vel_mag_pred)

    # ---------------- PLOTTING ----------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # ... (Codice di tripcolor e colorbar identico a prima) ...
    # Esempio:
    cols = ['Ground Truth', 'Prediction', 'Absolute Error']
    for ax, col in zip(axs[0], cols): ax.set_title(col, fontsize=16, pad=10)
    im0 = axs[0, 0].tripcolor(triang, vel_mag_gt, shading='gouraud', cmap='turbo')
    axs[0, 0].set_ylabel("Velocity Magnitude [m/s]", fontsize=14)
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
    # ... ripeti per gli altri 5 plot ...
    im1 = axs[0, 1].tripcolor(triang, vel_mag_pred, shading='gouraud', cmap='turbo')
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    im2 = axs[0, 2].tripcolor(triang, vel_err, shading='gouraud', cmap='inferno')
    fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)
    im3 = axs[1, 0].tripcolor(triang, T_gt, shading='gouraud', cmap='inferno')
    axs[1, 0].set_ylabel("Temperature [K]", fontsize=14)
    fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)
    im4 = axs[1, 1].tripcolor(triang, T_pred, shading='gouraud', cmap='inferno')
    fig.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)
    im5 = axs[1, 2].tripcolor(triang, T_err, shading='gouraud', cmap='inferno')
    fig.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)

    # Formattazione assi
    for ax in axs.flat:
        # FONDAMENTALE: 'equal' costringe matplotlib a rispettare le unità fisiche
        # Se X va da 0 a 2.5 e Y da 0 a 0.5, il plot sarà rettangolare.
        ax.set_aspect('equal')
        ax.axis('off')

    filename = f"prediction_epoch_{epoch}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close(fig)

def train_and_evaluate():
    # ----------------CONFIGURATION----------------
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create results directory
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "geofno_best_model.pth")

    print(f"Using device: {device}")
    print(f"Saving results to: {save_dir}/")

    # ----------------DATASET----------------
    # (Assuming dataset class is defined)
    full_dataset = MeshToGridDataset(
        root_dir="./../data_generation/dataset",
        grid_size=(64, 64),
        input_keys=["conductivity", "power"],
        output_keys=["temperature", "pressure", "vx", "vy"],
    )

    # Split Train (80%) / Test (20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )

    # ----------------MODEL----------------
    model = GeoFNO(modes1=12, modes2=12, width=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Scheduler to reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # History Logging
    train_loss_history = []
    test_error_history = []
    best_test_error = float('inf')

    print("Start Training...")
    print(f"{'Epoch':<5} | {'Train Loss':<12} | {'Test L2 Error':<15} | {'Status':<10}")
    print("-" * 50)

    for epoch in range(EPOCHS):
        # --- TRAINING ---
        model.train()
        train_loss_accum = 0

        for batch in train_loader:
            x_grid = batch["x_grid"].to(device, non_blocking=True)
            y_mesh = batch["y_mesh"].to(device, non_blocking=True)
            coords = batch["query_coords"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_mesh = model(x_grid, coords)

            # Masked Loss
            loss = compute_masked_loss(pred_mesh, y_mesh, mask)

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # --- VALIDATION / TEST ---
        model.eval()
        test_l2_accum = 0

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                x_grid = batch["x_grid"].to(device, non_blocking=True)
                y_mesh = batch["y_mesh"].to(device, non_blocking=True)
                coords = batch["query_coords"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)

                pred_mesh = model(x_grid, coords)

                # Relative L2 Error
                l2_err = compute_relative_l2(pred_mesh, y_mesh, mask)
                test_l2_accum += l2_err.item()

                # Visualize first batch of test set every 10 epochs
                if i == 0 and (epoch % 10 == 0 or epoch == EPOCHS - 1):
                    visualize_sample(coords, y_mesh, pred_mesh, mask, epoch, save_dir)

        avg_test_l2 = test_l2_accum / len(test_loader)
        test_error_history.append(avg_test_l2)

        # Step Scheduler
        scheduler.step(avg_test_l2)

        # --- LOGGING & SAVING ---
        status = ""
        if avg_test_l2 < best_test_error:
            best_test_error = avg_test_l2
            torch.save(model.state_dict(), model_save_path)
            status = "SAVED"

        print(f"{epoch + 1:<5} | {avg_train_loss:.6f}     | {avg_test_l2:.6f}        | {status}")

    # --- FINAL PLOTS ---
    plot_history(train_loss_history, test_error_history, save_dir)

    print("-" * 50)
    print(f"Training Complete. Best Test L2 Error: {best_test_error:.6f}")
    print(f"Model and plots saved in: {save_dir}")


if __name__ == "__main__":
    train_and_evaluate()