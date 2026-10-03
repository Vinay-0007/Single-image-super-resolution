import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import DIV2KDataset
from src.model import EDSR


# --------------------------------------------------
# Configuration
# --------------------------------------------------

HR_DIR = "data/DIV2K/DIV2K_train_HR"

BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

NUM_WORKERS = 0

CHECKPOINT_DIR = "checkpoints"

TRAIN_RATIO = 0.9


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Image split
# --------------------------------------------------

all_images = sorted([
    f for f in os.listdir(HR_DIR)
    if f.lower().endswith(".png")
])

random.seed(42)
random.shuffle(all_images)

split_index = int(len(all_images) * TRAIN_RATIO)

train_images = all_images[:split_index]
val_images = all_images[split_index:]

print("Total images:", len(all_images))
print("Training images:", len(train_images))
print("Validation images:", len(val_images))


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = DIV2KDataset(
    HR_DIR,
    train_images,
    patch_size=192,
    scale=4,
    train=True
)

val_dataset = DIV2KDataset(
    HR_DIR,
    val_images,
    patch_size=192,
    scale=4,
    train=False
)


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = EDSR(
    channels=64,
    num_blocks=16,
    scale=4
).to(device)


# --------------------------------------------------
# Loss and optimizer
# --------------------------------------------------

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.5
)


# --------------------------------------------------
# Mixed precision
# --------------------------------------------------

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# --------------------------------------------------
# Best PSNR
# --------------------------------------------------

best_psnr = 0.0


# --------------------------------------------------
# PSNR function
# --------------------------------------------------

def calculate_psnr(sr, hr):

    mse = torch.mean(
        (sr - hr) ** 2
    )

    if mse == 0:
        return float("inf")

    psnr = 10 * torch.log10(
        1.0 / mse
    )

    return psnr.item()


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(NUM_EPOCHS):

    # ==============================================
    # TRAINING
    # ==============================================

    model.train()

    train_loss = 0.0

    for batch_idx, (lr, hr) in enumerate(train_loader):

        lr = lr.to(
            device,
            non_blocking=True
        )

        hr = hr.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            sr = model(lr)

            loss = criterion(
                sr,
                hr
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:

            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    train_loss /= len(train_loader)


    # ==============================================
    # VALIDATION
    # ==============================================

    model.eval()

    val_loss = 0.0
    val_psnr = 0.0

    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(
                device,
                non_blocking=True
            )

            hr = hr.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                sr = model(lr)

                loss = criterion(
                    sr,
                    hr
                )

            val_loss += loss.item()

            # PSNR should use full precision
            sr = torch.clamp(
                sr.float(),
                0,
                1
            )

            batch_psnr = calculate_psnr(
                sr,
                hr.float()
            )

            val_psnr += batch_psnr

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)


    # ==============================================
    # Learning rate
    # ==============================================

    current_lr = optimizer.param_groups[0]["lr"]


    print("\n================================")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss : {train_loss:.6f}")
    print(f"Val Loss   : {val_loss:.6f}")
    print(f"Val PSNR   : {val_psnr:.2f} dB")
    print(f"Learning Rate: {current_lr:.2e}")
    print("================================")


    # ==============================================
    # Save latest
    # ==============================================

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_psnr": val_psnr
        },
        f"{CHECKPOINT_DIR}/edsr_epoch_{epoch+1}.pth"
    )


    # ==============================================
    # Save best by PSNR
    # ==============================================

    if val_psnr > best_psnr:

        best_psnr = val_psnr

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_psnr": val_psnr
            },
            f"{CHECKPOINT_DIR}/edsr_best.pth"
        )

        print(
            f"⭐ New best model! "
            f"PSNR: {val_psnr:.2f} dB"
        )


    # Update scheduler
    scheduler.step()


print("\nTraining complete!")
print(f"Best validation PSNR: {best_psnr:.2f} dB")