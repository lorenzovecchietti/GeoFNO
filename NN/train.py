"""
Main training script for the GeoFNO model.
Handles dataset loading, model initialization, training loop, and evaluation.
"""

import os
import time

import torch
from loader import MeshToGridDataset
from model import GeoFNO
from torch import optim
from torch.utils.data import DataLoader, random_split
from utils import (
    augment_batch,
    collate_fn,
    compute_masked_loss,
    visualize_sample, plot_history,
)

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
PATIENCE = 20
NUM_EXAMPLES = 10

# Loss weights for multi-task learning
W_TEMP = 1.0  # Weight for temperature loss (L2 relative)
W_VEL = 1.0   # Weight for velocity loss (L1)


# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def train_and_evaluate():
    """
    Sets up and runs the training and evaluation loop for GeoFNO with early stopping.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading dataset...")
    full_dataset = MeshToGridDataset(
        root_dir="./../data_generation/dataset",
        grid_size=(128, 128),
        input_keys=["conductivity", "power"],
        output_keys=["temperature", "vx", "vy"],
        force_recompute=True,  # Recompute to include vel_inlet channels
    )

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = GeoFNO(
        modes1=16,
        modes2=32,
        width=256,
        dropout_rate=0.1
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    best_test_error = float("inf")
    patience_counter = 0
    train_hist, test_hist = [], []

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x_grid = batch["x_grid"].to(device)
            y_mesh = batch["y_mesh"].to(device)
            coords = batch["query_coords"].to(device)
            mask = batch["mask"].to(device)

            # Data Augmentation
            x_grid, y_mesh, coords = augment_batch(x_grid, y_mesh, coords)

            optimizer.zero_grad()
            out = model(x_grid, coords)
            loss = compute_masked_loss(out, y_mesh, mask, w_temp=W_TEMP, w_vel=W_VEL)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        avg_train = train_loss / len(train_loader)
        train_hist.append(avg_train)

        # --- Validation ---
        model.eval()
        test_err = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                x_grid = batch["x_grid"].to(device)
                y_mesh = batch["y_mesh"].to(device)
                coords = batch["query_coords"].to(device)
                mask = batch["mask"].to(device)

                out = model(x_grid, coords)
                test_err += compute_masked_loss(out, y_mesh, mask, w_temp=W_TEMP, w_vel=W_VEL).item()

        avg_test = test_err / len(test_loader)
        test_hist.append(avg_test)

        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]

        # --- Early Stopping Logic ---
        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | Time: {epoch_time:.1f}s |"
            + f" Train: {avg_train:.5f} | Test: {avg_test:.5f} |"
            + f" Best: {best_test_error:.5f} | LR: {current_lr:.2e}"
        )

        if avg_test < best_test_error:
            best_test_error = avg_test
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print(f"\n--- EARLY STOPPING TRIGGERED at Epoch {epoch+1} ---")
                break

    print("Training complete.")

    # --- FINAL TEST SET VISUALIZATION ---
    print("\nStarting generation of Test Set visualizations...")

    best_model_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model weights.")
    else:
        print("Warning: Best model not found. Using last epoch weights.")

    model.eval()

    count = 0
    test_vis_dir = os.path.join(save_dir, "test_examples")
    os.makedirs(test_vis_dir, exist_ok=True)
    plot_history(train_hist, test_hist, save_dir)
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            if count >= NUM_EXAMPLES:
                break

            x_grid = batch["x_grid"].to(device)
            y_mesh = batch["y_mesh"].to(device)
            coords = batch["query_coords"].to(device)
            mask = batch["mask"].to(device)

            out = model(x_grid, coords)

            batch_size_curr = x_grid.shape[0]
            for i in range(batch_size_curr):
                if count >= NUM_EXAMPLES:
                    break

                print(f"Saving test example {count+1}/{NUM_EXAMPLES}...")
                visualize_sample(
                    x_grid,
                    coords,
                    y_mesh,
                    out,
                    mask,
                    epoch=None,
                    save_dir=test_vis_dir,
                    sample_idx=i,
                    filename_prefix=f"test_sample_{count:03d}",
                )
                count += 1

    print(f"Saved {count} test examples in {test_vis_dir}")


if __name__ == "__main__":
    train_and_evaluate()
