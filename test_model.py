import torch
from src.model import EDSR


# Use GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Create model
model = EDSR(
    channels=64,
    num_blocks=16,
    scale=4
).to(device)

# Dummy LR image
x = torch.randn(
    1, 3, 48, 48
).to(device)

# Forward pass
with torch.no_grad():
    y = model(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)

# Number of parameters
params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Parameters:",
    f"{params:,}"
)