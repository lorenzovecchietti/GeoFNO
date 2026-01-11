"""
Utility functions for GeoFNO training, including data collation,
loss metrics, visualization, and data augmentation.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import tri


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized mesh data in batches.
    Pads meshes to the maximum size in the batch and creates a mask.
    """
    max_n = max(item["y_mesh"].shape[0] for item in batch)
    x_grids, y_meshes, coords, masks = [], [], [], []

    for item in batch:
        x_grids.append(item["x_grid"])
        y, c = item["y_mesh"], item["query_coords"]
        n = y.shape[0]
        padding_len = max_n - n

        mask = torch.cat([torch.ones(n), torch.zeros(padding_len)], dim=0)
        masks.append(mask)

        if padding_len > 0:
            y_pad = torch.cat([y, torch.zeros(padding_len, y.shape[1])], dim=0)
            c_pad = torch.cat([c, torch.zeros(padding_len, c.shape[1])], dim=0)
        else:
            y_pad, c_pad = y, c

        y_meshes.append(y_pad)
        coords.append(c_pad)

    return {
        "x_grid": torch.stack(x_grids),
        "y_mesh": torch.stack(y_meshes),
        "query_coords": torch.stack(coords),
        "mask": torch.stack(masks),
    }


def compute_masked_loss(pred, target, mask):
    """
    Computes relative L2 loss (standard for FNO).
    Loss = ||Pred - Target||_2 / ||Target||_2
    """
    mask_expanded = mask.unsqueeze(-1)
    pred = pred * mask_expanded
    target = target * mask_expanded

    diff_norms = torch.norm(pred - target, p=2, dim=(1, 2))
    target_norms = torch.norm(target, p=2, dim=(1, 2))
    loss = (diff_norms / (target_norms + 1e-8)).mean()

    return loss


# pylint: disable=too-many-arguments, too-many-positional-arguments
# pylint: disable=too-many-locals, too-many-statements
def visualize_sample(
    x_grid,
    coords,
    target,
    pred,
    mask,
    epoch,
    save_dir,
    sample_idx=None,
    filename_prefix="prediction",
    len_x=20,
    len_y=5,
):
    """
    Generates FEM-style visualization plots.
    If sample_idx is None, selects a random sample from the batch.
    If sample_idx is specified, plots that specific batch element.
    """
    if sample_idx is None:
        i = np.random.randint(0, len(x_grid))
    else:
        i = sample_idx

    # Data Preparation
    c_norm = coords[i].cpu().numpy()
    t = target[i].cpu().numpy()
    p = pred[i].cpu().numpy()
    m = mask[i].cpu().numpy()
    x_g = x_grid[i].cpu().numpy()  # Shape: (C, H, W)

    # Filter out padding
    valid_indices = m == 1
    c_valid = c_norm[valid_indices]
    t_vals = t[valid_indices]
    p_vals = p[valid_indices]

    # 2. Geometric Denormalization
    c_01 = (c_valid + 1) / 2.0
    phys_x = c_01[:, 0] * len_x
    phys_y = c_01[:, 1] * len_y

    # 3. Mesh Reconstruction
    triang = tri.Triangulation(phys_x, phys_y)

    # 4. Physical Field Extraction
    t_gt = t_vals[:, 0]
    vel_mag_gt = np.sqrt(t_vals[:, 2] ** 2 + t_vals[:, 3] ** 2)
    t_pred = p_vals[:, 0]
    vel_mag_pred = np.sqrt(p_vals[:, 2] ** 2 + p_vals[:, 3] ** 2)
    t_err = np.abs(t_gt - t_pred)
    vel_err = np.abs(vel_mag_gt - vel_mag_pred)

    # Compute shared colorbar scales
    vel_min = min(vel_mag_gt.min(), vel_mag_pred.min())
    vel_max = max(vel_mag_gt.max(), vel_mag_pred.max())
    t_min = min(t_gt.min(), t_pred.min())
    t_max = max(t_gt.max(), t_pred.max())

    # Extract input fields
    conductivity = x_g[0]
    power = x_g[1]
    domain_mask = x_g[4]

    conductivity_masked = np.where(domain_mask > 0.5, conductivity, np.nan)
    power_masked = np.where(domain_mask > 0.5, power, np.nan)

    # 5. Plotting
    fig, axs = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)

    cols = ["Ground Truth", "Prediction", "Absolute Error"]
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontsize=16, pad=10)

    # Velocity Magnitude plots
    im0 = axs[0, 0].tripcolor(
        triang, vel_mag_gt, shading="gouraud", cmap="turbo", vmin=vel_min, vmax=vel_max
    )
    axs[0, 0].set_ylabel("Velocity Magnitude [m/s]", fontsize=14)
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].tripcolor(
        triang,
        vel_mag_pred,
        shading="gouraud",
        cmap="turbo",
        vmin=vel_min,
        vmax=vel_max,
    )
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = axs[0, 2].tripcolor(triang, vel_err, shading="gouraud", cmap="inferno")
    fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    # Temperature plots
    im3 = axs[1, 0].tripcolor(
        triang, t_gt, shading="gouraud", cmap="inferno", vmin=t_min, vmax=t_max
    )
    axs[1, 0].set_ylabel("Temperature [K]", fontsize=14)
    fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im4 = axs[1, 1].tripcolor(
        triang, t_pred, shading="gouraud", cmap="inferno", vmin=t_min, vmax=t_max
    )
    fig.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

    im5 = axs[1, 2].tripcolor(triang, t_err, shading="gouraud", cmap="inferno")
    fig.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)

    # Input fields plots
    axs[2, 0].set_title("Conductivity (Input)", fontsize=16, pad=10)
    im6 = axs[2, 0].imshow(
        conductivity_masked,
        extent=[0, len_x, 0, len_y],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    axs[2, 0].set_ylabel("Input Fields", fontsize=14)
    fig.colorbar(im6, ax=axs[2, 0], fraction=0.046, pad=0.04)

    axs[2, 1].set_title("Power (Input)", fontsize=16, pad=10)
    im7 = axs[2, 1].imshow(
        power_masked,
        extent=[0, len_x, 0, len_y],
        origin="lower",
        cmap="hot",
        aspect="equal",
    )
    fig.colorbar(im7, ax=axs[2, 1], fraction=0.046, pad=0.04)

    axs[2, 2].set_title("Domain Mask", fontsize=16, pad=10)
    im8 = axs[2, 2].imshow(
        domain_mask,
        extent=[0, len_x, 0, len_y],
        origin="lower",
        cmap="gray",
        aspect="equal",
    )
    fig.colorbar(im8, ax=axs[2, 2], fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.set_aspect("equal")
        ax.axis("off")

    if epoch is not None:
        fname = f"{filename_prefix}_epoch_{epoch}.png"
    else:
        fname = f"{filename_prefix}.png"

    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)


def plot_history(train_losses, test_errors, save_dir):
    """
    Plots the training loss and test error history.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss (MSE)")
    plt.yscale("log")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(test_errors, color="orange", label="Test Error (Rel L2)")
    plt.yscale("log")
    plt.title("Test Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"))
    plt.close()


def augment_batch(x_grid, y_mesh, coords):
    """
    Applies random flips and rotations (0, 90, 180, 270 degrees) for data augmentation.
    """
    # Random horizontal flip
    if torch.rand(1) < 0.5:
        x_grid = x_grid.flip(2)
        coords = coords.clone()
        coords[..., 1] = -coords[..., 1]
        y_mesh = y_mesh.clone()
        y_mesh[..., 3] = -y_mesh[..., 3]  # Flip vy component

    # Random rotation in 90-degree increments
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        x_grid = torch.rot90(x_grid, k, [2, 3])

        # Apply rotation matrix to coordinates
        rad = k * (np.pi / 2)
        rot_matrix = torch.tensor(
            [[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]],
            device=coords.device,
            dtype=coords.dtype,
        )

        coords = torch.matmul(coords, rot_matrix.T)

        # Rotate velocity vectors (vx, vy at indices 2, 3)
        vel = y_mesh[..., 2:4]
        vel_rot = torch.matmul(vel, rot_matrix.T)
        y_mesh = y_mesh.clone()
        y_mesh[..., 2:4] = vel_rot

    return x_grid, y_mesh, coords
