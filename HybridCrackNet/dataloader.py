import torch
import torchvision
from dataset import CrackDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

mu = [0.51789941, 0.51360926, 0.547762]
sd  = [0.1812099,  0.17746663, 0.20386334]
train_transform = A.Compose(
    [
        A.Normalize(
            mean=mu,
            std=sd,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

val_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=mu,
            std=sd,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, test_dir, test_maskdir, batch_size, num_workers=4, pin_memory=True,):
    # train dataset
    train_ds = CrackDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform,)
    train_loader = DataLoader(train_ds, batch_size=batch_size, pin_memory=pin_memory, shuffle=True, num_workers=4)
    # validation dataset
    val_ds = CrackDataset(image_dir=val_dir,mask_dir=val_maskdir,transform=train_transform,)
    val_loader = DataLoader(val_ds,batch_size=batch_size,pin_memory=pin_memory,shuffle=False, num_workers=4)
    # test dataset
    test_ds = CrackDataset(image_dir=test_dir,mask_dir=test_maskdir,transform=train_transform,)
    test_loader = DataLoader(test_ds,batch_size=batch_size,pin_memory=pin_memory,shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
