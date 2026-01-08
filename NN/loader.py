import torch
from torch.utils.data import Dataset
import numpy as np
import meshio
import os
from scipy.interpolate import griddata

class MeshToGridDataset(Dataset):
    def __init__(self, root_dir, grid_size=(64, 64), normalize=True):
        self.root_dir = root_dir
        self.grid_size = grid_size
        self.normalize = normalize
        self.cases = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
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

    def _compute_or_load_stats(self):
        """Calcola (o carica) media e std per normalizzare i dati."""
        stats_path = os.path.join(self.root_dir, 'stats.pt')
        
        if os.path.exists(stats_path):
            print(f"Caricamento statistiche da {stats_path}...")
            stats = torch.load(stats_path)
            self.input_mean = stats['input_mean']
            self.input_std = stats['input_std']
            self.output_mean = stats['output_mean']
            self.output_std = stats['output_std']
        else:
            print("Calcolo statistiche sul dataset (può richiedere tempo)...")
            # Accumulatori
            in_data_list = []
            out_data_list = []
            
            # Leggiamo tutti i file per calcolare le statistiche globali
            for case in self.cases:
                input_path = os.path.join(self.root_dir, case, 'input.npy')
                output_path = os.path.join(self.root_dir, case, 'output.npy')
                
                in_data_list.append(np.load(input_path)) # (N, 2)
                out_data_list.append(np.load(output_path)) # (N, 4)
            
            # Concateniamo tutto in un unico grande array per calcolare mean/std
            all_inputs = np.concatenate(in_data_list, axis=0)
            all_outputs = np.concatenate(out_data_list, axis=0)
            
            self.input_mean = torch.tensor(all_inputs.mean(axis=0), dtype=torch.float32)
            self.input_std = torch.tensor(all_inputs.std(axis=0), dtype=torch.float32) + 1e-8 # epsilon
            
            self.output_mean = torch.tensor(all_outputs.mean(axis=0), dtype=torch.float32)
            self.output_std = torch.tensor(all_outputs.std(axis=0), dtype=torch.float32) + 1e-8 # epsilon
            
            # Salviamo per la prossima volta
            torch.save({
                'input_mean': self.input_mean, 'input_std': self.input_std,
                'output_mean': self.output_mean, 'output_std': self.output_std
            }, stats_path)
            print("Statistiche calcolate e salvate.")

    def denormalize_output(self, tensor):
        """Utile per convertire le predizioni del modello in valori reali."""
        if not self.normalize:
            return tensor
        # tensor shape: (..., 4)
        return tensor * self.output_std.to(tensor.device) + self.output_mean.to(tensor.device)

    def __getitem__(self, idx):
        case_path = os.path.join(self.root_dir, self.cases[idx])
        
        # 1. Caricamento Dati
        # Mesh: forza lettura solo prime 2 colonne (x,y) anche se c'è z
        mesh = meshio.read(os.path.join(case_path, 'mesh.msh'))
        points = mesh.points[:, :2].astype(np.float32) 
        
        input_nodes = np.load(os.path.join(case_path, 'input.npy')).astype(np.float32)
        output_nodes = np.load(os.path.join(case_path, 'output.npy')).astype(np.float32)

        # 2. Normalizzazione Dati (Se attiva)
        if self.normalize:
            input_nodes = (torch.tensor(input_nodes) - self.input_mean) / self.input_std
            output_nodes = (torch.tensor(output_nodes) - self.output_mean) / self.output_std
            input_nodes = input_nodes.numpy()
            output_nodes = output_nodes.numpy()

        # 3. Normalizzazione Coordinate (0 to 1) per GridData
        min_pos = points.min(axis=0)
        max_pos = points.max(axis=0)
        # Aggiungiamo 1e-8 per evitare divisione per zero
        norm_points = (points - min_pos) / (max_pos - min_pos + 1e-8) 
        
        # 4. Interpolazione Input su Griglia
        # Nota: interpoliamo i dati GIA' normalizzati. 
        # fill_value=0 è ok perché i dati sono normalizzati (media ~0)
        grid_k = griddata(norm_points, input_nodes[:, 0], (self.grid_x, self.grid_y), method='linear', fill_value=0)
        grid_Q = griddata(norm_points, input_nodes[:, 1], (self.grid_x, self.grid_y), method='linear', fill_value=0)
        
        # Maschera binaria (Geometria)
        grid_mask = griddata(norm_points, np.ones(len(points)), (self.grid_x, self.grid_y), method='nearest', fill_value=0)
        
        # Stack Input (5 canali)
        input_grid = np.stack([
            grid_k, 
            grid_Q, 
            self.grid_x, 
            self.grid_y, 
            grid_mask
        ], axis=0) # (5, H, W)
        
        # 5. Query Coords per GeoFNO
        # GridSample vuole coordinate in [-1, 1]
        query_coords = norm_points * 2 - 1 
        
        return {
            'x_grid': torch.from_numpy(input_grid).float(),
            'y_mesh': torch.from_numpy(output_nodes).float(), # Output Normalizzato
            'query_coords': torch.from_numpy(query_coords).float()
        }