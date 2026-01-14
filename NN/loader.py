"""
Module for loading mesh data and interpolating it onto a regular grid for FNO.
Revised to avoid library noise and optimize training speed using Multiprocessing.
"""

import contextlib
import functools
import io
import logging
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor

import meshio
import numpy as np
import torch
from scipy.interpolate import griddata
from torch.utils.data import Dataset
from tqdm import tqdm  # Recommended for progress tracking

# Absolute silence for libraries that talk too much
logging.getLogger("meshio").setLevel(logging.ERROR)
os.environ["MESHIO_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")


# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments
class MeshToGridDataset(Dataset):
    """
    Dataset class that converts mesh data to grid data with hybrid interpolation.
    Pre-loads everything to RAM using Multiprocessing.
    """

    def __init__(
        self,
        root_dir: str,
        grid_size: tuple = (128, 128),
        normalize: bool = True,
        input_keys: list | None = None,
        output_keys: list | None = None,
        force_recompute: bool = False,
        num_workers: int = 16,  # New argument for parallelism
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
            num_workers: Number of CPU processes to use for loading.
        """
        self.root_dir = root_dir
        self.grid_size = grid_size
        self.normalize = normalize
        self.input_keys = input_keys if input_keys is not None else ["conductivity"]
        self.output_keys = output_keys if output_keys is not None else ["solution"]
        self.force_recompute = force_recompute
        self.num_workers = num_workers

        self.cases = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )

        # Fixed grid in [0, 1] range (Used for reference, passed to workers)
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

    @staticmethod
    def _extract_data_from_dict_static(file_path: str, keys: list):
        """Static helper to extract data (must be static for pickling)."""
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

    # pylint: disable=too-many-locals, too-many-statements
    @staticmethod
    def _process_single_case(
        case_name,
        root_dir,
        grid_size,
        input_keys,
        output_keys,
        grid_x,
        grid_y,
        force_recompute,
        stats_dict,  # Pass stats as a dict/tuple, avoiding full self reference
        normalize,
    ):
        """
        Worker function to process a single case.
        Must be static and self-contained for multiprocessing.
        """
        # Re-silence inside worker process
        logging.getLogger("meshio").setLevel(logging.ERROR)

        case_path = os.path.join(root_dir, case_name)
        grid_cache_path = os.path.join(
            case_path, f"grid_hybrid_{grid_size[0]}x{grid_size[1]}.npy"
        )

        # 1. Load mesh points
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                mesh = meshio.read(os.path.join(case_path, "mesh.msh"))
                points = mesh.points[:, :2].astype(np.float32)
        except Exception:
            mesh = meshio.read(os.path.join(case_path, "mesh.msh"))
            points = mesh.points[:, :2].astype(np.float32)

        # 1b. Load vel_inlet
        params_path = os.path.join(case_path, "params.pkl")
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        vel_inlet = params["vel_inlet"]

        # 2. Extract solution data
        output_nodes = MeshToGridDataset._extract_data_from_dict_static(
            os.path.join(case_path, "solutions.npy"), output_keys
        ).astype(np.float32)

        # Normalize coordinates [-1, 1]
        min_pos = points.min(axis=0)
        max_pos = points.max(axis=0)
        norm_points = (points - min_pos) / (max_pos - min_pos + 1e-8)
        query_coords = norm_points * 2 - 1

        # 3. Process Input Grid
        if os.path.exists(grid_cache_path) and not force_recompute:
            input_grid = np.load(grid_cache_path)
        else:
            input_nodes_raw = MeshToGridDataset._extract_data_from_dict_static(
                os.path.join(case_path, "inputs.npy"), input_keys
            ).astype(np.float32)

            # Auto-mask detection
            try:
                k_idx = input_keys.index("k")
            except ValueError:
                k_idx = 0

            k_values = input_nodes_raw[:, k_idx]
            vals, counts = np.unique(np.round(k_values, 5), return_counts=True)
            k_fluid_detected = vals[np.argmax(counts)]
            node_mask_solid = (np.abs(k_values - k_fluid_detected) > 1e-2).astype(
                np.float32
            )

            # Apply Stats if normalized
            if normalize and stats_dict is not None:
                # Convert torch stats to numpy for the worker
                in_mean = stats_dict["input_mean"].numpy()
                in_std = stats_dict["input_std"].numpy()
                input_nodes = (input_nodes_raw - in_mean) / in_std
            else:
                input_nodes = input_nodes_raw

            grid_channels = []
            for i in range(input_nodes.shape[1]):
                # Linear Interp
                grid_c_linear = griddata(
                    norm_points,
                    input_nodes[:, i],
                    (grid_x, grid_y),
                    method="linear",
                    fill_value=np.nan,
                )
                # Nearest fallback
                mask_nan = np.isnan(grid_c_linear)
                if np.any(mask_nan):
                    grid_c_nearest = griddata(
                        norm_points,
                        input_nodes[:, i],
                        (grid_x, grid_y),
                        method="nearest",
                    )
                    grid_c_linear[mask_nan] = grid_c_nearest[mask_nan]
                grid_channels.append(grid_c_linear)

            # Mask Interp (Nearest)
            grid_mask = griddata(
                norm_points,
                node_mask_solid,
                (grid_x, grid_y),
                method="nearest",
                fill_value=0,
            )

            # Coordinate channels
            grid_channels.append(grid_x.astype(np.float32))
            grid_channels.append(grid_y.astype(np.float32))
            grid_channels.append(grid_mask.astype(np.float32))

            # Vel Inlet channel (single scalar value)
            vel_inlet_scalar = np.full_like(grid_x, vel_inlet, dtype=np.float32)
            grid_channels.append(vel_inlet_scalar)

            input_grid = np.stack(grid_channels, axis=0).astype(np.float32)
            np.save(grid_cache_path, input_grid)

        # 4. Final output normalization
        if normalize and stats_dict is not None:
            out_mean = stats_dict["output_mean"]
            out_std = stats_dict["output_std"]
            # Keep as numpy here, convert to tensor in main process to be safe/clean
            output_nodes_norm = (output_nodes - out_mean.numpy()) / out_std.numpy()
        else:
            output_nodes_norm = output_nodes

        # Return dict of numpy arrays (lighter to move across processes than tensors)
        return {
            "x_grid": input_grid,
            "y_mesh": output_nodes_norm,
            "query_coords": query_coords,
            "mask": np.ones(output_nodes_norm.shape[0]),
        }

    def _pre_load_data(self):
        """Pre-loads all dataset cases into RAM with pre-padding."""
        print(f"Pre-loading {len(self.cases)} cases with {self.num_workers} workers...")

        # Prepare stats dictionary to pass to workers
        stats_dict = None
        if self.normalize:
            stats_dict = {
                "input_mean": self.input_mean,
                "input_std": self.input_std,
                "output_mean": self.output_mean,
                "output_std": self.output_std,
            }

        # Partially apply arguments that are constant across all cases
        worker_func = functools.partial(
            self._process_single_case,
            root_dir=self.root_dir,
            grid_size=self.grid_size,
            input_keys=self.input_keys,
            output_keys=self.output_keys,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            force_recompute=self.force_recompute,
            stats_dict=stats_dict,
            normalize=self.normalize,
        )

        # Use ProcessPoolExecutor to run in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results_iter = list(
                tqdm(
                    executor.map(worker_func, self.cases),
                    total=len(self.cases),
                    desc="Loading Data",
                )
            )

        # Find max_nodes for pre-padding (eliminates dynamic padding in collate_fn)
        max_nodes = max(res["y_mesh"].shape[0] for res in results_iter)
        print(f"Pre-padding all samples to max_nodes={max_nodes}...")

        # Convert to Tensor and pre-pad in the main process
        for res in results_iter:
            n_nodes = res["y_mesh"].shape[0]
            pad_len = max_nodes - n_nodes

            # Create mask before padding
            mask = np.ones(max_nodes, dtype=np.float32)
            if pad_len > 0:
                mask[n_nodes:] = 0.0
                # Pad y_mesh and query_coords
                y_padded = np.pad(
                    res["y_mesh"], ((0, pad_len), (0, 0)), mode="constant"
                )
                c_padded = np.pad(
                    res["query_coords"], ((0, pad_len), (0, 0)), mode="constant"
                )
            else:
                y_padded = res["y_mesh"]
                c_padded = res["query_coords"]

            self.data_cache.append(
                {
                    "x_grid": torch.from_numpy(res["x_grid"]).float(),
                    "y_mesh": torch.from_numpy(y_padded).float(),
                    "query_coords": torch.from_numpy(c_padded).float(),
                    "mask": torch.from_numpy(mask).float(),
                }
            )

        self.max_nodes = max_nodes
        print(f"Data pre-loaded successfully. All samples padded to {max_nodes} nodes.")

    def _extract_data_from_dict(self, file_path: str, keys: list):
        """Instance method wrapper for the static method (legacy support)."""
        return self._extract_data_from_dict_static(file_path, keys)

    # _compute_or_load_stats REMAINS THE SAME AS ORIGINAL
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
                # Use the static method here too
                in_list.append(
                    self._extract_data_from_dict_static(in_path, self.input_keys)
                )
                out_list.append(
                    self._extract_data_from_dict_static(out_path, self.output_keys)
                )

            all_in = np.concatenate(in_list, axis=0)
            all_out = np.concatenate(out_list, axis=0)

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
