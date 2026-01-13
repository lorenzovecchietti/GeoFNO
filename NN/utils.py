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
    Fast collate function for pre-padded data.
    Since all samples are pre-padded during dataset initialization,
    this just stacks tensors without any dynamic padding.
    """
    return {
        "x_grid": torch.stack([item["x_grid"] for item in batch]),
        "y_mesh": torch.stack([item["y_mesh"] for item in batch]),
        "query_coords": torch.stack([item["query_coords"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }


def compute_masked_loss(pred, target, mask, w_temp=1.0, w_vel=1.0):
    """
    Computes relative L2 loss for temperature prediction.

    Args:
        pred: Temperature predictions (B, N, 1)
        target: Ground truth temperature (B, N, 1)
        mask: Valid node mask (B, N)
        w_temp: Weight for temperature loss (kept for API compatibility)
        w_vel: Not used (kept for API compatibility)

    Returns:
        Relative L2 loss for temperature
    """
    mask_expanded = mask.unsqueeze(-1)
    pred = pred * mask_expanded
    target = target * mask_expanded

    # Temperature: Relative L2 loss
    diff_T = torch.norm(pred - target, p=2, dim=(1, 2))
    norm_T = torch.norm(target, p=2, dim=(1, 2))
    loss_T = (diff_T / (norm_T + 1e-8)).mean()

    return loss_T



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

    # 4. Physical Field Extraction (output format: T only)
    t_gt = t_vals[:, 0]
    t_pred = p_vals[:, 0]
    
    # Compute relative error in percentage
    t_err_relative = np.abs((t_gt - t_pred) / (t_gt + 1e-8)) * 100.0
    
    # Compute shared colorbar scales for temperature
    t_min = min(t_gt.min(), t_pred.min())
    t_max = max(t_gt.max(), t_pred.max())

    # Extract input fields
    conductivity = x_g[0]
    power = x_g[1]
    domain_mask = x_g[4]

    conductivity_masked = np.where(domain_mask > 0.5, conductivity, np.nan)
    power_masked = np.where(domain_mask > 0.5, power, np.nan)

    # 5. Plotting (2x3 grid: Temperature row + Input fields row)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # Temperature row
    axs[0, 0].set_title("Ground Truth Temperature", fontsize=16, pad=10)
    im0 = axs[0, 0].tripcolor(
        triang, t_gt, shading="gouraud", cmap="inferno", vmin=t_min, vmax=t_max
    )
    axs[0, 0].set_ylabel("Temperature [K]", fontsize=14)
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    axs[0, 1].set_title("Predicted Temperature", fontsize=16, pad=10)
    im1 = axs[0, 1].tripcolor(
        triang, t_pred, shading="gouraud", cmap="inferno", vmin=t_min, vmax=t_max
    )
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    axs[0, 2].set_title("Relative Error (%)", fontsize=16, pad=10)
    im2 = axs[0, 2].tripcolor(
        triang, t_err_relative, shading="gouraud", cmap="hot", vmin=0, vmax=10
    )
    fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04, label='Error %')

    # Input fields row
    axs[1, 0].set_title("Conductivity (Input)", fontsize=16, pad=10)
    im3 = axs[1, 0].imshow(
        conductivity_masked,
        extent=[0, len_x, 0, len_y],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    axs[1, 0].set_ylabel("Input Fields", fontsize=14)
    fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

    axs[1, 1].set_title("Power (Input)", fontsize=16, pad=10)
    im4 = axs[1, 1].imshow(
        power_masked,
        extent=[0, len_x, 0, len_y],
        origin="lower",
        cmap="hot",
        aspect="equal",
    )
    fig.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

    axs[1, 2].set_title("Domain Mask", fontsize=16, pad=10)
    im5 = axs[1, 2].imshow(
        domain_mask,
        extent=[0, len_x, 0, len_y],
        origin="lower",
        cmap="gray",
        aspect="equal",
    )
    fig.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)

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
    Optimized to clone only once at the start to minimize memory allocations.

    x_grid channels: [conductivity, power, x, y, mask, vel_inlet]
    y_mesh channels: [T]
    """
    do_flip = torch.rand(1).item() < 0.5
    k = torch.randint(0, 4, (1,)).item()

    # Early return if no augmentation needed
    if not do_flip and k == 0:
        return x_grid, y_mesh, coords

    # Clone once at the start (only if we need to modify)
    x_grid = x_grid.clone()
    y_mesh = y_mesh.clone()
    coords = coords.clone()

    # Horizontal flip
    if do_flip:
        x_grid = x_grid.flip(2)
        coords[..., 1] = -coords[..., 1]
        # No velocity field to flip in y_mesh (only temperature)

    # Rotation in 90-degree increments
    if k > 0:
        x_grid = torch.rot90(x_grid, k, [2, 3])

        # Precompute rotation matrix
        rad = k * (np.pi / 2)
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        rot_matrix = torch.tensor(
            [[cos_r, -sin_r], [sin_r, cos_r]],
            device=coords.device,
            dtype=coords.dtype,
        )

        # Rotate coordinates
        coords = torch.matmul(coords, rot_matrix.T)
        # No velocity vectors to rotate (vel_inlet is scalar, y_mesh is temperature only)

    return x_grid, y_mesh, coords

