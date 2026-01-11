"""
Module for loading mesh data and interpolating it onto a regular grid for FNO.
Revised to avoid library noise and optimize training speed.
"""

import contextlib
import io
import logging
import os
import warnings

import meshio
import numpy as np
import torch
from scipy.interpolate import griddata
from torch.utils.data import Dataset

# Absolute silence for libraries that talk too much
logging.getLogger("meshio").setLevel(logging.ERROR)
os.environ["MESHIO_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments
class MeshToGridDataset(Dataset):
    """
    Dataset class that converts mesh data to grid data with hybrid interpolation.
    Pre-loads everything to RAM to ensure silent and fast training.
    """

    def __init__(
        self,
        root_dir: str,
        grid_size: tuple = (128, 128),
        normalize: bool = True,
        input_keys: list | None = None,
        output_keys: list | None = None,
        force_recompute: bool = False,
    ):
        """
        Initializes the dataset.

        Args:
            root_dir: Path to the dataset root folder.
            grid_size: Resolution of the regular grid for FNO.
            normalize: Whether to apply Z-score normalization.
            input_keys: List of keys to extract from input files.
            output_keys: List of keys to extract from solution files.
            force_recompute: If True, recomputes grid interpolation even if cached.
        """
        self.root_dir = root_dir
        self.grid_size = grid_size
        self.normalize = normalize
        self.input_keys = input_keys if input_keys is not None else ["conductivity"]
        self.output_keys = output_keys if output_keys is not None else ["solution"]
        self.force_recompute = force_recompute

        self.cases = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )

        # Fixed grid in [0, 1] range
        xi = np.linspace(0, 1, grid_size[1])
        yi = np.linspace(0, 1, grid_size[0])
        self.grid_x, self.grid_y = np.meshgrid(xi, yi)

        # Stats initialization
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

        if self.normalize:
            self._compute_or_load_stats()

        # Data cache for fast memory access
        self.data_cache: list[dict[str, np.ndarray]] = []
        self._pre_load_data()

    # pylint: disable=too-many-locals, too-many-statements
    def _pre_load_data(self):
        """Pre-loads all dataset cases into RAM to speed up training."""

        print(f"Pre-loading {len(self.cases)} cases into RAM...")
        for _, case_name in enumerate(self.cases):
            case_path = os.path.join(self.root_dir, case_name)
            grid_cache_path = os.path.join(
                case_path, f"grid_hybrid_{self.grid_size[0]}x{self.grid_size[1]}.npy"
            )

            # 1. Load mesh points for ground truth coordinates
            try:
                with contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(io.StringIO()):
                    mesh = meshio.read(os.path.join(case_path, "mesh.msh"))
                    points = mesh.points[:, :2].astype(np.float32)
            except Exception:  # pylint: disable=broad-exception-caught
                mesh = meshio.read(os.path.join(case_path, "mesh.msh"))
                points = mesh.points[:, :2].astype(np.float32)

            # 2. Extract solution data on mesh nodes
            output_nodes = self._extract_data_from_dict(
                os.path.join(case_path, "solutions.npy"), self.output_keys
            ).astype(np.float32)

            # Normalize coordinates to range [-1, 1] for grid_sample
            min_pos = points.min(axis=0)
            max_pos = points.max(axis=0)
            norm_points = (points - min_pos) / (max_pos - min_pos + 1e-8)
            query_coords = norm_points * 2 - 1

            # 3. Process Input Grid
            if os.path.exists(grid_cache_path) and not self.force_recompute:
                input_grid = np.load(grid_cache_path)
            else:
                # Load RAW input nodes
                input_nodes_raw = self._extract_data_from_dict(
                    os.path.join(case_path, "inputs.npy"), self.input_keys
                ).astype(np.float32)

                # Automatic fluid/solid mask detection based on conductivity
                try:
                    k_idx = self.input_keys.index("k")
                except ValueError:
                    k_idx = 0

                k_values = input_nodes_raw[:, k_idx]

                # Detect fluid as the most frequent conductivity value
                vals, counts = np.unique(np.round(k_values, 5), return_counts=True)
                k_fluid_detected = vals[np.argmax(counts)]

                # Create solid mask: 1.0 for solid, 0.0 for fluid
                epsilon = 1e-2
                node_mask_solid = (
                    np.abs(k_values - k_fluid_detected) > epsilon
                ).astype(np.float32)

                # Apply normalization to inputs if requested
                if self.normalize:
                    input_nodes = (
                        input_nodes_raw - self.input_mean.numpy()
                    ) / self.input_std.numpy()
                else:
                    input_nodes = input_nodes_raw

                grid_channels = []
                for i in range(input_nodes.shape[1]):
                    # Hybrid interpolation: linear with nearest-neighbor fallback
                    grid_c_linear = griddata(
                        norm_points,
                        input_nodes[:, i],
                        (self.grid_x, self.grid_y),
                        method="linear",
                        fill_value=np.nan,
                    )
                    mask_nan = np.isnan(grid_c_linear)
                    if np.any(mask_nan):
                        grid_c_nearest = griddata(
                            norm_points,
                            input_nodes[:, i],
                            (self.grid_x, self.grid_y),
                            method="nearest",
                        )
                        grid_c_linear[mask_nan] = grid_c_nearest[mask_nan]
                    grid_channels.append(grid_c_linear)

                # Nearest-neighbor for sharp boundaries
                grid_mask = griddata(
                    norm_points,
                    node_mask_solid,
                    (self.grid_x, self.grid_y),
                    method="nearest",
                    fill_value=0,
                )

                # Append coordinates and mask as additional input channels
                grid_channels.append(self.grid_x.astype(np.float32))
                grid_channels.append(self.grid_y.astype(np.float32))
                grid_channels.append(grid_mask.astype(np.float32))

                input_grid = np.stack(grid_channels, axis=0).astype(np.float32)
                np.save(grid_cache_path, input_grid)

            # 4. Final normalization and tensor conversion
            if self.normalize:
                output_nodes_t = (
                    torch.tensor(output_nodes) - self.output_mean
                ) / self.output_std
            else:
                output_nodes_t = torch.tensor(output_nodes)

            self.data_cache.append(
                {
                    "x_grid": torch.from_numpy(input_grid).float(),
                    "y_mesh": output_nodes_t.float(),
                    "query_coords": torch.from_numpy(query_coords).float(),
                    "mask": torch.ones(output_nodes_t.shape[0]),
                }
            )
        print("Data pre-loaded successfully.")

    def _extract_data_from_dict(self, file_path: str, keys: list):
        """Loads a .npy file and extracts requested keys as a NumPy array."""
        data_obj = np.load(file_path, allow_pickle=True)
        if data_obj.ndim == 0:
            data_dict = data_obj.item()
        elif (
            isinstance(data_obj, np.ndarray)
            and len(data_obj) == 1
            and isinstance(data_obj[0], dict)
        ):
            data_dict = data_obj[0]
        else:
            data_dict = data_obj

        extracted_arrays = []
        for key in keys:
            if key in data_dict:
                val = data_dict[key]
                if val.ndim == 1:
                    val = val[:, np.newaxis]
                extracted_arrays.append(val)
            else:
                raise KeyError(f"Key '{key}' not found in {file_path}")
        return np.concatenate(extracted_arrays, axis=1)

    def _compute_or_load_stats(self):
        """Computes or loads normalization statistics (mean and std)."""
        stats_path = os.path.join(self.root_dir, "stats.pt")
        if os.path.exists(stats_path) and not self.force_recompute:
            stats = torch.load(stats_path)
            self.input_mean = stats["input_mean"]
            self.input_std = stats["input_std"]
            self.output_mean = stats["output_mean"]
            self.output_std = stats["output_std"]
        else:
            print("Computing dataset statistics...")
            in_list, out_list = [], []
            for case in self.cases:
                in_path = os.path.join(self.root_dir, case, "inputs.npy")
                out_path = os.path.join(self.root_dir, case, "solutions.npy")
                if not os.path.exists(in_path) or not os.path.exists(out_path):
                    continue
                in_list.append(self._extract_data_from_dict(in_path, self.input_keys))
                out_list.append(
                    self._extract_data_from_dict(out_path, self.output_keys)
                )

            all_in, all_out = np.concatenate(in_list, axis=0), np.concatenate(
                out_list, axis=0
            )
            self.input_mean = torch.tensor(all_in.mean(axis=0), dtype=torch.float32)
            self.input_std = (
                torch.tensor(all_in.std(axis=0), dtype=torch.float32) + 1e-8
            )
            self.output_mean = torch.tensor(all_out.mean(axis=0), dtype=torch.float32)
            self.output_std = (
                torch.tensor(all_out.std(axis=0), dtype=torch.float32) + 1e-8
            )
            torch.save(
                {
                    "input_mean": self.input_mean,
                    "input_std": self.input_std,
                    "output_mean": self.output_mean,
                    "output_std": self.output_std,
                },
                stats_path,
            )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        return self.data_cache[idx]
