import os
import torch
import numpy as np

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from swinir.network_swinir import SwinIR


CHECKPOINT = (
    "swinir/models/"
    "001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
)

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
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ==================================================
# Load SwinIR
# ==================================================

model = SwinIR(
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
    CHECKPOINT,
    map_location=device
)

# Official classical SR checkpoint
# stores weights under "params"
if "params" in checkpoint:
    checkpoint = checkpoint["params"]

model.load_state_dict(
    checkpoint,
    strict=True
)

model = model.to(device)
model.eval()

print("SwinIR model loaded successfully.")


# ==================================================
# Evaluate one dataset
# ==================================================

def evaluate_dataset(
    dataset_name,
    dataset_path
):

    print("\n================================")
    print(dataset_name)
    print("================================")

    files = sorted([
        f
        for f in os.listdir(dataset_path)
        if f.endswith("_LR.png")
    ])

    print("Images:", len(files))

    psnr_values = []
    ssim_values = []

    with torch.no_grad():

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

            # ----------------------------------
            # Load images
            # ----------------------------------

            lr_image = Image.open(
                lr_path
            ).convert("RGB")

            hr_image = Image.open(
                hr_path
            ).convert("RGB")

            # ----------------------------------
            # LR → tensor
            # ----------------------------------

            lr = np.asarray(
                lr_image
            ).astype(
                np.float32
            ) / 255.0

            lr = torch.from_numpy(
                lr
            )

            lr = lr.permute(
                2, 0, 1
            )

            lr = lr.unsqueeze(0)

            lr = lr.to(device)

            # ----------------------------------
            # SwinIR
            # ----------------------------------

            sr = model(lr)

            sr = torch.clamp(
                sr,
                0,
                1
            )

            sr = sr.squeeze(0)

            sr = sr.permute(
                1, 2, 0
            )

            sr = sr.cpu().numpy()

            # ----------------------------------
            # HR → numpy
            # ----------------------------------

            hr = np.asarray(
                hr_image
            ).astype(
                np.float32
            ) / 255.0

            # ----------------------------------
            # Match dimensions
            # ----------------------------------

            h = min(
                sr.shape[0],
                hr.shape[0]
            )

            w = min(
                sr.shape[1],
                hr.shape[1]
            )

            sr = sr[:h, :w]
            hr = hr[:h, :w]

            # ----------------------------------
            # PSNR
            # ----------------------------------

            psnr = peak_signal_noise_ratio(
                hr,
                sr,
                data_range=1.0
            )

            # ----------------------------------
            # SSIM
            # ----------------------------------

            ssim = structural_similarity(
                hr,
                sr,
                channel_axis=2,
                data_range=1.0
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim)

    average_psnr = np.mean(
        psnr_values
    )

    average_ssim = np.mean(
        ssim_values
    )

    print(
        f"Average PSNR: "
        f"{average_psnr:.2f} dB"
    )

    print(
        f"Average SSIM: "
        f"{average_ssim:.4f}"
    )

    return (
        average_psnr,
        average_ssim
    )


# ==================================================
# Run all datasets
# ==================================================

results = {}

for dataset_name, dataset_path in DATASETS.items():

    results[dataset_name] = evaluate_dataset(
        dataset_name,
        dataset_path
    )


# ==================================================
# Final results
# ==================================================

print("\n\n==============================================")
print("FINAL SwinIR ×4 BENCHMARK RESULTS")
print("==============================================")

print(
    f"{'Dataset':<12}"
    f"{'PSNR (dB)':>15}"
    f"{'SSIM':>15}"
)

print("----------------------------------------------")

for dataset_name, (psnr, ssim) in results.items():

    print(
        f"{dataset_name:<12}"
        f"{psnr:>15.2f}"
        f"{ssim:>15.4f}"
    )

print("==============================================")