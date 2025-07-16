import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

PATCH_SIZE   = 7          # receptive field of one cortical column
OVERLAP      = 0.5        # overlap ratio between patches
MINICOLS     = 64         # minicolumns per column
NEUR_PER_MIN = 16         # neurons per minicolumn
BATCH        = 128
EPOCHS       = 6
LR           = 1e-3
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

def image_to_patches(x: torch.Tensor, patch_size: int, overlap_ratio: float):
    """x: (B,1,H,W)  ->  (B, N_patches, 1, patch_size, patch_size)"""
    stride = int(patch_size * (1 - overlap_ratio))
    patches = F.unfold(x, kernel_size=patch_size, stride=stride)
    L = patches.size(-1)
    patches = patches.permute(0, 2, 1).reshape(x.size(0), L, 1, patch_size, patch_size)
    return patches

class CorticalColumn(nn.Module):
    """One column = MINICOLS parallel 1×1 convs (minicolumns) -> intra-column vote"""
    def __init__(self, channels: int = 1):
        super().__init__()
        self.minicolumns = nn.ModuleList([
            nn.Conv2d(channels, NEUR_PER_MIN, kernel_size=1)
            for _ in range(MINICOLS)
        ])

    def forward(self, x):
        votes = torch.stack([m(x) for m in self.minicolumns], dim=1)   # (B,MINICOLS,NEUR_PER_MIN,P,P)
        consensus = votes.mean(dim=1)                                  # (B,NEUR_PER_MIN,P,P)
        return consensus.view(consensus.size(0), -1)                   # flatten

class ThousandsBrainsNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.column = CorticalColumn()
        self.classifier = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        patches = image_to_patches(x, PATCH_SIZE, OVERLAP)  # (B,L,1,P,P)
        descriptors = []
        for i in range(patches.size(1)):
            desc = self.column(patches[:, i])               # (B, F_local)
            descriptors.append(desc)
        global_desc = torch.stack(descriptors, dim=1).mean(dim=1)  # (B, F)
        return self.classifier(global_desc)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST(root='.', train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root='.', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

if __name__ == "__main__":
    model = ThousandsBrainsNet().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    history = {'loss': [], 'acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # -- train --
        model.train()
        tot = ok = 0
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            ok  += (logits.argmax(1) == y).sum().item()
            tot += x.size(0)
        history['loss'].append(running_loss / tot)
        history['acc'].append(ok / tot)

        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                ok  += (logits.argmax(1) == y).sum().item()
                tot += x.size(0)
        history['val_acc'].append(ok / tot)
        print(f'Epoch {epoch+1}  val_acc={ok/tot:.4f}')

    plt.figure(figsize=(6,3))
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['acc'], label='train acc')
    plt.plot(history['val_acc'], label='val acc')
    plt.legend(); plt.xlabel('epoch'); plt.title('TBT on MNIST'); plt.tight_layout(); plt.show()

    filters = model.column.minicolumns[0].weight.detach().cpu().squeeze()
    plt.figure(figsize=(8,2))
    for i in range(min(16, filters.size(0))):
        plt.subplot(2,8,i+1)
        plt.imshow(filters[i:i+1].unsqueeze(0), cmap='coolwarm', vmin=-1, vmax=1)
        plt.axis('off')
    plt.suptitle('First minicolumn 1×1 filters'); plt.show()

    model.eval()
    x, y = test_ds[0]
    x_in = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        patches = image_to_patches(x_in, PATCH_SIZE, OVERLAP)
        votes = [(model.column(patches[:, i]).norm(dim=1)).item() for i in range(patches.size(1))]
    side = int(math.sqrt(len(votes)))
    if side * side == len(votes):
        plt.figure(figsize=(4,4))
        plt.imshow(torch.tensor(votes).reshape(side,side), cmap='viridis')
        plt.colorbar(label='activation'); plt.title(f'Vote map for digit {y}'); plt.show()