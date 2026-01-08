"""
Module for loading mesh data and interpolating it onto a regular grid for FNO.
"""

import os

import meshio
import numpy as np
import torch
from scipy.interpolate import griddata
from torch.utils.data import Dataset


class MeshToGridDataset(Dataset):
    """
    Dataset class that converts mesh data to grid data.
    """

    def __init__(
        self,
        root_dir,
        grid_size=(64, 64),
        normalize=True,
        input_keys=None,
        output_keys=None,
    ):
        """
        Args:
            root_dir (str): Cartella dati.
            grid_size (tuple): Dimensione target della griglia.
            normalize (bool): Se normalizzare i dati.
            input_keys (list): Lista delle chiavi da estrarre dal dizionario di input.
            output_keys (list): Lista delle chiavi da estrarre dal dizionario di output.
        """
        self.root_dir = root_dir
        self.grid_size = grid_size
        self.normalize = normalize
        self.input_keys = input_keys if input_keys is not None else ["conductivity"]
        self.output_keys = output_keys if output_keys is not None else ["solution"]

        self.cases = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )

        # Grid fissa [0,1]
        xi = np.linspace(0, 1, grid_size[1])
        yi = np.linspace(0, 1, grid_size[0])
        self.grid_x, self.grid_y = np.meshgrid(xi, yi)

        # Inizializzazione statistiche
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

        if self.normalize:
            self._compute_or_load_stats()

    def _extract_data_from_dict(self, file_path, keys):
        """Carica il file .npy ed estrae le chiavi richieste."""
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
                raise KeyError(
                    f"Chiave '{key}' non trovata nel file {file_path}. "
                    f"Chiavi: {data_dict.keys()}"
                )

        return np.concatenate(extracted_arrays, axis=1)

    def _compute_or_load_stats(self):
        """Calcola (o carica) media e std per normalizzare i dati."""
        stats_path = os.path.join(self.root_dir, "stats.pt")

        if os.path.exists(stats_path):
            print(f"Caricamento statistiche da {stats_path}...")
            stats = torch.load(stats_path)
            self.input_mean = stats["input_mean"]
            self.input_std = stats["input_std"]
            self.output_mean = stats["output_mean"]
            self.output_std = stats["output_std"]
        else:
            print("Calcolo statistiche sul dataset (può richiedere tempo)...")
            in_data_list = []
            out_data_list = []

            for case in self.cases:
                input_path = os.path.join(self.root_dir, case, "inputs.npy")
                output_path = os.path.join(self.root_dir, case, "solutions.npy")
                in_data = self._extract_data_from_dict(input_path, self.input_keys)
                out_data = self._extract_data_from_dict(output_path, self.output_keys)
                in_data_list.append(in_data)
                out_data_list.append(out_data)

            all_inputs = np.concatenate(in_data_list, axis=0)
            all_outputs = np.concatenate(out_data_list, axis=0)

            self.input_mean = torch.tensor(all_inputs.mean(axis=0), dtype=torch.float32)
            self.input_std = (
                torch.tensor(all_inputs.std(axis=0), dtype=torch.float32) + 1e-8
            )
            self.output_mean = torch.tensor(
                all_outputs.mean(axis=0), dtype=torch.float32
            )
            self.output_std = (
                torch.tensor(all_outputs.std(axis=0), dtype=torch.float32) + 1e-8
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
            print("Statistiche calcolate e salvate.")

    def denormalize_output(self, tensor):
        """
        Denormalize the output tensor using stored statistics.
        """
        if not self.normalize:
            return tensor
        return tensor * self.output_std.to(tensor.device) + self.output_mean.to(
            tensor.device
        )

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_name = self.cases[idx]
        case_path = os.path.join(self.root_dir, case_name)

        # Grid cache path
        grid_cache_path = os.path.join(
            case_path, f"grid_{self.grid_size[0]}x{self.grid_size[1]}.npy"
        )

        # 1. Caricamento Mesh (sempre necessario per query_coords)
        mesh = meshio.read(os.path.join(case_path, "mesh.msh"))
        points = mesh.points[:, :2].astype(np.float32)

        # 2. Caricamento Dati Output e Query Coords
        output_nodes = self._extract_data_from_dict(
            os.path.join(case_path, "solutions.npy"), self.output_keys
        ).astype(np.float32)

        # Normalizzazione Coordinate Spaziali
        min_pos = points.min(axis=0)
        max_pos = points.max(axis=0)
        norm_points = (points - min_pos) / (max_pos - min_pos + 1e-8)
        query_coords = norm_points * 2 - 1

        # 3. Gestione Cache per Input Grid
        if os.path.exists(grid_cache_path):
            input_grid = np.load(grid_cache_path)
        else:
            # Caricamento Dati Input per Interpolazione
            input_nodes = self._extract_data_from_dict(
                os.path.join(case_path, "inputs.npy"), self.input_keys
            ).astype(np.float32)

            # Normalizzazione Input
            if self.normalize:
                # Usiamo torch per comodità visto che mean/std sono tensor
                input_nodes_t = (
                    torch.tensor(input_nodes) - self.input_mean
                ) / self.input_std
                input_nodes = input_nodes_t.numpy()

            # Interpolazione Input su Griglia
            grid_channels = []
            for i in range(input_nodes.shape[1]):
                grid_c = griddata(
                    norm_points,
                    input_nodes[:, i],
                    (self.grid_x, self.grid_y),
                    method="linear",
                    fill_value=0,
                )
                grid_channels.append(grid_c)

            grid_mask = griddata(
                norm_points,
                np.ones(len(points)),
                (self.grid_x, self.grid_y),
                method="nearest",
                fill_value=0,
            )

            grid_channels.append(self.grid_x)
            grid_channels.append(self.grid_y)
            grid_channels.append(grid_mask)
            input_grid = np.stack(grid_channels, axis=0).astype(np.float32)

            # Salvataggio in Cache
            np.save(grid_cache_path, input_grid)

        # 4. Normalizzazione Output
        if self.normalize:
            output_nodes = (
                torch.tensor(output_nodes) - self.output_mean
            ) / self.output_std
        else:
            output_nodes = torch.tensor(output_nodes)

        return {
            "x_grid": torch.from_numpy(input_grid).float(),
            "y_mesh": output_nodes.float(),
            "query_coords": torch.from_numpy(query_coords).float(),
        }
