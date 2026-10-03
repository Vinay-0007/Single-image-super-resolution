import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from src.model import EDSR
from esrgan.RRDBNet_arch import RRDBNet
from swinir.network_swinir import SwinIR


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# IMAGE
# ============================================================

lr_path = "data/Set5/image_SRF_4/img_001_SRF_4_LR.png"
hr_path = "data/Set5/image_SRF_4/img_001_SRF_4_HR.png"


lr_image = Image.open(
    lr_path
).convert("RGB")

hr_image = Image.open(
    hr_path
).convert("RGB")


# ============================================================
# IMAGE → TENSOR
# ============================================================

def image_to_tensor(image):

    image = np.asarray(
        image
    ).astype(
        np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        image
    )

    tensor = tensor.permute(
        2, 0, 1
    )

    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


lr = image_to_tensor(
    lr_image
)


# ============================================================
# LOAD EDSR
# ============================================================

edsr = EDSR(
    channels=64,
    num_blocks=16,
    scale=4
).to(device)

checkpoint = torch.load(
    "checkpoints/edsr_best.pth",
    map_location=device
)

edsr.load_state_dict(
    checkpoint["model_state_dict"]
)

edsr.eval()

print("EDSR loaded.")


# ============================================================
# LOAD ESRGAN
# ============================================================

esrgan = RRDBNet(
    in_nc=3,
    out_nc=3,
    nf=64,
    nb=23,
    gc=32,
    sf=4
).to(device)

checkpoint = torch.load(
    "esrgan/models/RRDB_ESRGAN_x4.pth",
    map_location=device
)

esrgan.load_state_dict(
    checkpoint,
    strict=True
)

esrgan.eval()

print("ESRGAN loaded.")


# ============================================================
# LOAD SWINIR
# ============================================================

swinir = SwinIR(
    upscale=4,
    in_chans=3,
    img_size=48,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv"
)

checkpoint = torch.load(
    "swinir/models/"
    "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth",
    map_location=device
)

if "params" in checkpoint:
    checkpoint = checkpoint["params"]

swinir.load_state_dict(
    checkpoint,
    strict=True
)

swinir = swinir.to(device)
swinir.eval()

print("SwinIR loaded.")


# ============================================================
# GENERATE SUPER-RESOLVED IMAGES
# ============================================================

with torch.no_grad():

    edsr_output = edsr(lr)

    esrgan_output = esrgan(lr)

    swinir_output = swinir(lr)


# ============================================================
# CONVERT TENSOR → IMAGE
# ============================================================

def tensor_to_image(tensor):

    tensor = torch.clamp(
        tensor,
        0,
        1
    )

    tensor = tensor.squeeze(0)

    tensor = tensor.permute(
        1,
        2,
        0
    )

    tensor = tensor.cpu().numpy()

    return tensor


edsr_image = tensor_to_image(
    edsr_output
)

esrgan_image = tensor_to_image(
    esrgan_output
)

swinir_image = tensor_to_image(
    swinir_output
)

lr_display = np.asarray(
    lr_image
).astype(
    np.float32
) / 255.0

hr_display = np.asarray(
    hr_image
).astype(
    np.float32
) / 255.0


# ============================================================
# VISUAL COMPARISON
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(20, 5)
)


axes[0].imshow(
    lr_display
)

axes[0].set_title(
    "LR Input\n48 × 48"
)


axes[1].imshow(
    edsr_image
)

axes[1].set_title(
    "EDSR\n192 × 192"
)


axes[2].imshow(
    esrgan_image
)

axes[2].set_title(
    "ESRGAN\n192 × 192"
)


axes[3].imshow(
    swinir_image
)

axes[3].set_title(
    "SwinIR\n192 × 192"
)


axes[4].imshow(
    hr_display
)

axes[4].set_title(
    "HR Ground Truth\n192 × 192"
)


for ax in axes:

    ax.axis("off")


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    "visual_comparison_Set5.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nVisual comparison saved as:"
)

print(
    "visual_comparison_Set5.png"
)