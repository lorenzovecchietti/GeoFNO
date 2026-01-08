import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class GeoFNO(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(GeoFNO, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        # Input: k, Q, x, y, mask
        self.fc0 = nn.Linear(5, self.width) 
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 4) # Output: T, u, v, P

    def forward(self, x_grid, query_coords):
        # 1. Processing sulla Griglia Latente
        # x_grid shape: (Batch, 5, H, W)
        x = x_grid.permute(0, 2, 3, 1) # (B, H, W, C)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)

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
        x = x1 + x2 # Output Latente su Griglia (B, width, H, W)

        # 2. Querying Geometrica (Grid -> Mesh)
        # Vogliamo i valori non su tutta la griglia, ma sui nodi esatti della mesh.
        # query_coords ha shape (B, N_points, 2) ma grid_sample vuole (B, 1, N_points, 2)
        # Trattiamo i punti come una "immagine" di altezza 1 e larghezza N
        q = query_coords.unsqueeze(1) 
        
        # grid_sample campiona il tensore 'x' alle coordinate 'q'.
        # mode='bilinear' interpola dolcemente i valori della griglia.
        x_sampled = F.grid_sample(x, q, mode='bilinear', padding_mode='zeros', align_corners=False)
        # x_sampled shape: (B, width, 1, N_points)
        
        x_sampled = x_sampled.squeeze(2).permute(0, 2, 1) # (B, N_points, width)

        # 3. Decoder finale (point-wise)
        x_sampled = self.fc1(x_sampled)
        x_sampled = F.gelu(x_sampled)
        out = self.fc2(x_sampled) # (B, N_points, 4)
        
        return out

def train():
    # Setup
    dataset = MeshToGridDataset(root_dir='./tuoi_dati', grid_size=(64, 64))
    # Batch size 1 è spesso necessario se ogni mesh ha un numero diverso di nodi (N variabile).
    # Se tutte le mesh hanno lo stesso N, puoi usare batch_size > 1.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) 
    
    model = GeoFNO(modes1=12, modes2=12, width=32).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("Start Training GeoFNO...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            x_grid = batch['x_grid'].cuda()           # (B, 5, H, W)
            y_mesh = batch['y_mesh'].cuda()           # (B, N, 4)
            coords = batch['query_coords'].cuda()     # (B, N, 2)
            
            optimizer.zero_grad()
            
            # Forward: Input Grid -> FNO -> Sample at Coords -> Output Mesh Values
            pred_mesh = model(x_grid, coords)
            
            # Loss calcolata direttamente sui nodi
            loss = loss_fn(pred_mesh, y_mesh)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss/len(dataloader):.5f}")

if __name__ == "__main__":
    train()