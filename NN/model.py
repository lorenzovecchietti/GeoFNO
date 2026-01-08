"""
Fourier Neural Operator implementation for Geometric FNO.
"""

import torch
import torch.nn.functional as F
from loader import MeshToGridDataset
from torch import cfloat, einsum, fft, nn, optim, rand
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")


def collate_fn(batch):
    """
    Funzione statica per gestire batch di mesh con numero variabile di nodi.
    Da usare nel DataLoader: collate_fn=MeshToGridDataset.collate_fn
    """
    # 1. Trova la dimensione massima N in questo batch
    max_n = max(item["y_mesh"].shape[0] for item in batch)

    x_grids = []
    y_meshes = []
    coords = []
    masks = []

    for item in batch:
        # x_grid è fissa (C, H, W), basta aggiungerla alla lista
        x_grids.append(item["x_grid"])

        # Recuperiamo i dati variabili
        y = item["y_mesh"]  # (N, out_channels)
        c = item["query_coords"]  # (N, 2)
        n = y.shape[0]

        padding_len = max_n - n

        # --- Creazione Mask ---
        # 1 = Dato reale, 0 = Padding
        mask = torch.cat([torch.ones(n), torch.zeros(padding_len)], dim=0)
        masks.append(mask)

        # --- Padding Dati ---
        if padding_len > 0:
            # Padding con zeri in fondo
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
        # 1. Processing sulla Griglia Latente
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
        x = x1 + x2  # Output Latente su Griglia (B, width, H, W)

        # 2. Querying Geometrica (Grid -> Mesh)
        # Vogliamo i valori non su tutta la griglia, ma sui nodi esatti della mesh.
        # query_coords ha shape (B, N_points, 2) ma grid_sample vuole (B, 1, N_points, 2)
        # Trattiamo i punti come una "immagine" di altezza 1 e larghezza N
        q = query_coords.unsqueeze(1)

        # grid_sample campiona il tensore 'x' alle coordinate 'q'.
        # mode='bilinear' interpola dolcemente i valori della griglia.
        x_sampled = F.grid_sample(
            x, q, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        # x_sampled shape: (B, width, 1, N_points)

        x_sampled = x_sampled.squeeze(2).permute(0, 2, 1)  # (B, N_points, width)

        # 3. Decoder finale (point-wise)
        x_sampled = self.fc1(x_sampled)
        x_sampled = F.gelu(x_sampled)
        out = self.fc2(x_sampled)  # (B, N_points, 4)

        return out


def train():
    """
    Training loop for GeoFNO.
    """
    # Setup
    # Quando crei il dataset nel tuo file main o di training:
    dataset = MeshToGridDataset(
        root_dir="./../data_generation/dataset",
        grid_size=(64, 64),
        # Input: usa la chiave vista nella prima immagine
        input_keys=["conductivity", "power"],
        # Output: usa le chiavi che il messaggio di errore ti ha elencato
        output_keys=["temperature", "pressure", "vx", "vy"],
    )
    # Batch size 1 è spesso necessario se ogni mesh ha un numero diverso di nodi (N variabile).
    # Se tutte le mesh hanno lo stesso N, puoi usare batch_size > 1.
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = GeoFNO(modes1=12, modes2=12, width=32).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Start Training (Standard Float32)...")

    for epoch in range(100):
        model.train()
        total_loss = 0

        for batch in dataloader:
            x_grid = batch["x_grid"].cuda(non_blocking=True)
            y_mesh = batch["y_mesh"].cuda(non_blocking=True)
            coords = batch["query_coords"].cuda(non_blocking=True)

            optimizer.zero_grad()

            pred_mesh = model(x_grid, coords)
            loss = F.mse_loss(pred_mesh, y_mesh)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss {total_loss / len(dataloader):.5f}")


if __name__ == "__main__":
    train()
