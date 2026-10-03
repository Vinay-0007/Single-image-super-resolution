import matplotlib.pyplot as plt
import numpy as np

datasets = ["Set5", "Set14", "BSD100", "Urban100"]

# ==============================
# PSNR
# ==============================

edsr_psnr = [27.19, 24.61, 25.03, 22.13]
esrgan_psnr = [28.44, 24.34, 23.97, 22.79]
swinir_psnr = [30.72, 26.87, 26.54, 25.50]

x = np.arange(len(datasets))
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(
    x - width,
    edsr_psnr,
    width,
    label="EDSR"
)

plt.bar(
    x,
    esrgan_psnr,
    width,
    label="ESRGAN"
)

plt.bar(
    x + width,
    swinir_psnr,
    width,
    label="SwinIR"
)

plt.xlabel("Dataset")
plt.ylabel("PSNR (dB)")
plt.title("PSNR Comparison of EDSR, ESRGAN and SwinIR")
plt.xticks(x, datasets)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "PSNR_comparison.png",
    dpi=300
)

plt.show()


# ==============================
# SSIM
# ==============================

edsr_ssim = [0.8042, 0.7074, 0.6911, 0.6767]
esrgan_ssim = [0.8310, 0.6790, 0.6471, 0.7219]
swinir_ssim = [0.8851, 0.7753, 0.7452, 0.8111]

plt.figure(figsize=(10, 6))

plt.bar(
    x - width,
    edsr_ssim,
    width,
    label="EDSR"
)

plt.bar(
    x,
    esrgan_ssim,
    width,
    label="ESRGAN"
)

plt.bar(
    x + width,
    swinir_ssim,
    width,
    label="SwinIR"
)

plt.xlabel("Dataset")
plt.ylabel("SSIM")
plt.title("SSIM Comparison of EDSR, ESRGAN and SwinIR")
plt.xticks(x, datasets)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "SSIM_comparison.png",
    dpi=300
)

plt.show()


# ==============================
# LPIPS
# ==============================

edsr_lpips = [0.3113, 0.4185, 0.4902, 0.4411]
esrgan_lpips = [0.0750, 0.1337, 0.1615, 0.1230]
swinir_lpips = [0.1685, 0.2698, 0.3590, 0.1934]

plt.figure(figsize=(10, 6))

plt.bar(
    x - width,
    edsr_lpips,
    width,
    label="EDSR"
)

plt.bar(
    x,
    esrgan_lpips,
    width,
    label="ESRGAN"
)

plt.bar(
    x + width,
    swinir_lpips,
    width,
    label="SwinIR"
)

plt.xlabel("Dataset")
plt.ylabel("LPIPS")
plt.title("LPIPS Comparison of EDSR, ESRGAN and SwinIR")
plt.xticks(x, datasets)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "LPIPS_comparison.png",
    dpi=300
)

plt.show()

print("Graphs generated successfully!")