import matplotlib.pyplot as plt
from src.dataset import DIV2KDataset


hr_dir = "data/DIV2K/DIV2K_train_HR"

dataset = DIV2KDataset(hr_dir)

lr, hr = dataset[0]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(lr.permute(1, 2, 0))
plt.title("LR - 48×48")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hr.permute(1, 2, 0))
plt.title("HR - 192×192")
plt.axis("off")

plt.tight_layout()
plt.show()