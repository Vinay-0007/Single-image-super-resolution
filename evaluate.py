import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from src.model import EDSR


# --------------------------------------------------
# Configuration
# --------------------------------------------------

HR_DIR = "data/DIV2K/DIV2K_train_HR"
CHECKPOINT = "checkpoints/edsr_best.pth"

PATCH_SIZE = 192
SCALE = 4

TRAIN_RATIO = 0.9
NUM_VISUALS = 5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Recreate the same 720/80 split
# --------------------------------------------------

all_images = sorted([
    f for f in os.listdir(HR_DIR)
    if f.lower().endswith(".png")
])

random.seed(42)
random.shuffle(all_images)

split_index = int(
    len(all_images) * TRAIN_RATIO
)

val_images = all_images[split_index:]

print("Validation images:", len(val_images))


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = EDSR(
    channels=64,
    num_blocks=16,
    scale=4
).to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Loaded best model from epoch:",
    checkpoint["epoch"]
)


# --------------------------------------------------
# Results directory
# --------------------------------------------------

os.makedirs(
    "results/validation",
    exist_ok=True
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

psnr_values = []
ssim_values = []


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

with torch.no_grad():

    for i, filename in enumerate(val_images):

        path = os.path.join(
            HR_DIR,
            filename
        )

        # -----------------------------
        # Load HR
        # -----------------------------

        hr_image = Image.open(
            path
        ).convert("RGB")

        width, height = hr_image.size

        # Center crop
        x = (width - PATCH_SIZE) // 2
        y = (height - PATCH_SIZE) // 2

        hr_image = hr_image.crop(
            (
                x,
                y,
                x + PATCH_SIZE,
                y + PATCH_SIZE
            )
        )


        # -----------------------------
        # Create LR
        # -----------------------------

        lr_image = hr_image.resize(
            (
                PATCH_SIZE // SCALE,
                PATCH_SIZE // SCALE
            ),
            Image.Resampling.BICUBIC
        )


        # -----------------------------
        # Convert LR to tensor
        # -----------------------------

        lr = np.array(
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


        # -----------------------------
        # EDSR
        # -----------------------------

        sr = model(lr)

        sr = torch.clamp(
            sr,
            0,
            1
        )

        sr = sr.squeeze(
            0
        )

        sr = sr.permute(
            1, 2, 0
        )

        sr = sr.cpu().numpy()


        # -----------------------------
        # HR numpy
        # -----------------------------

        hr = np.array(
            hr_image
        ).astype(
            np.float32
        ) / 255.0


        # -----------------------------
        # PSNR
        # -----------------------------

        psnr = peak_signal_noise_ratio(
            hr,
            sr,
            data_range=1.0
        )


        # -----------------------------
        # SSIM
        # -----------------------------

        ssim = structural_similarity(
            hr,
            sr,
            channel_axis=2,
            data_range=1.0
        )


        psnr_values.append(
            psnr
        )

        ssim_values.append(
            ssim
        )


        print(
            f"{filename} | "
            f"PSNR: {psnr:.2f} dB | "
            f"SSIM: {ssim:.4f}"
        )


        # -----------------------------
        # Save visual examples
        # -----------------------------

        if i < NUM_VISUALS:

            plt.figure(
                figsize=(12, 4)
            )

            plt.subplot(
                1, 3, 1
            )

            plt.imshow(
                lr_image
            )

            plt.title(
                "LR 48×48"
            )

            plt.axis(
                "off"
            )


            plt.subplot(
                1, 3, 2
            )

            plt.imshow(
                sr
            )

            plt.title(
                f"EDSR\nPSNR={psnr:.2f} dB"
            )

            plt.axis(
                "off"
            )


            plt.subplot(
                1, 3, 3
            )

            plt.imshow(
                hr
            )

            plt.title(
                "HR 192×192"
            )

            plt.axis(
                "off"
            )


            plt.tight_layout()

            output_path = (
                f"results/validation/"
                f"comparison_{i+1}.png"
            )

            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close()


# --------------------------------------------------
# Final results
# --------------------------------------------------

average_psnr = np.mean(
    psnr_values
)

average_ssim = np.mean(
    ssim_values
)


print("\n================================")
print("FINAL EDSR VALIDATION RESULTS")
print("================================")

print(
    f"Images evaluated: {len(val_images)}"
)

print(
    f"Average PSNR: {average_psnr:.2f} dB"
)

print(
    f"Average SSIM: {average_ssim:.4f}"
)

print("================================")