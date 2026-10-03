import os
import torch
import numpy as np
import lpips

from PIL import Image

from src.model import EDSR
from esrgan.RRDBNet_arch import RRDBNet
from swinir.network_swinir import SwinIR


# ============================================================
# Configuration
# ============================================================

DATASETS = {
    "Set5": "data/Set5/image_SRF_4",
    "Set14": "data/Set14/image_SRF_4",
    "BSD100": "data/BSD100/image_SRF_4",
    "Urban100": "data/Urban100/image_SRF_4"
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LPIPS model
# ============================================================

loss_fn = lpips.LPIPS(net="alex").to(device)
loss_fn.eval()

print("LPIPS model loaded.")


# ============================================================
# Load EDSR
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
# Load ESRGAN
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
# Load SwinIR
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
# Convert PIL image to tensor
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


# ============================================================
# Calculate LPIPS
# ============================================================

def calculate_lpips(model, lr, hr):

    with torch.no_grad():

        sr = model(lr)

        sr = torch.clamp(
            sr,
            0,
            1
        )

        # LPIPS expects [-1, 1]
        sr = sr * 2 - 1
        hr = hr * 2 - 1

        score = loss_fn(
            sr,
            hr
        )

        return score.item()


# ============================================================
# Evaluate model on dataset
# ============================================================

def evaluate_model(
    model,
    dataset_path
):

    files = sorted([
        f
        for f in os.listdir(dataset_path)
        if f.endswith("_LR.png")
    ])

    scores = []

    for filename in files:

        lr_path = os.path.join(
            dataset_path,
            filename
        )

        hr_filename = filename.replace(
            "_LR.png",
            "_HR.png"
        )

        hr_path = os.path.join(
            dataset_path,
            hr_filename
        )

        if not os.path.exists(hr_path):
            continue

        lr_image = Image.open(
            lr_path
        ).convert("RGB")

        hr_image = Image.open(
            hr_path
        ).convert("RGB")

        lr = image_to_tensor(
            lr_image
        )

        hr = image_to_tensor(
            hr_image
        )

        score = calculate_lpips(
            model,
            lr,
            hr
        )

        scores.append(score)

    return np.mean(scores)


# ============================================================
# Run benchmark
# ============================================================

models = {
    "EDSR": edsr,
    "ESRGAN": esrgan,
    "SwinIR": swinir
}

results = {}


for dataset_name, dataset_path in DATASETS.items():

    print("\n================================")
    print(dataset_name)
    print("================================")

    results[dataset_name] = {}

    for model_name, model in models.items():

        print(
            f"Evaluating {model_name}..."
        )

        score = evaluate_model(
            model,
            dataset_path
        )

        results[dataset_name][model_name] = score

        print(
            f"{model_name} LPIPS: "
            f"{score:.4f}"
        )


# ============================================================
# Final table
# ============================================================

print("\n\n==============================================")
print("FINAL LPIPS RESULTS")
print("==============================================")

print(
    f"{'Dataset':<12}"
    f"{'EDSR':>15}"
    f"{'ESRGAN':>15}"
    f"{'SwinIR':>15}"
)

print("----------------------------------------------")

for dataset_name in DATASETS:

    print(
        f"{dataset_name:<12}"
        f"{results[dataset_name]['EDSR']:>15.4f}"
        f"{results[dataset_name]['ESRGAN']:>15.4f}"
        f"{results[dataset_name]['SwinIR']:>15.4f}"
    )

print("==============================================")