from src.dataset import DIV2KDataset


hr_dir = "data/DIV2K/DIV2K_train_HR"

dataset = DIV2KDataset(hr_dir)

print("Number of images:", len(dataset))

lr, hr = dataset[0]

print("LR shape:", lr.shape)
print("HR shape:", hr.shape)