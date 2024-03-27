import random
from functools import partial

# torchvision
import torchvision.transforms as transforms

# torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch

from utils import load_data, IMG_MEAN, IMG_STD


class PairImageDataset(Dataset):
    """
    Dataset that loads pairs of line & sketch images.
    
    When loading, apply given transformation and random (affine) augmentation.
    """

    def __init__(self, data_path, transform, cache_size: int|None = None):
        self.transform = transform
        self.line_images, self.sketch_images = load_data(data_path=data_path,
                                                         colorspace="L",
                                                         size=cache_size)

    def __len__(self):
        return len(self.line_images)

    def __getitem__(self, index):
        line_image, sketch_image = self.line_images[index], self.sketch_images[index]

        # Random affine augmentation.
        W, H = line_image.size
        angle = random.randint(-180, 180)
        translate = (random.randint(-W//2, W//2), random.randint(-H//2, H//2))
        tf_affine = partial(transforms.functional.affine,
                            angle=angle,
                            translate=translate,
                            scale=1.,
                            shear=0.,
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            fill=255)

        line_image, sketch_image = tf_affine(line_image), tf_affine(sketch_image)

        # Apply given transformation.
        line_image, sketch_image = transforms.functional.to_tensor(line_image), transforms.functional.to_tensor(sketch_image)
        images = torch.cat([line_image.unsqueeze(0), sketch_image.unsqueeze(0)], dim=0)
        images = self.transform(images)
        return images[0], images[1]


# class RandomResize(object):
#     """
#     Transformation that randomly resizes the image to one of the specified sizes.
#     """

#     def __init__(self, resizes):
#         self.resizes = resizes
    
#     def __call__(self, img):
#         return transforms.functional.resize(img, size=random.choice(self.resizes), antialias=True)


def get_dataloader(data_path: str, batch_size: int, img_size: int, cache_size: int|None = None,
                   num_workers: int = 0, split_ratio: float = 0.95):
    # Randomly resize, crop, and normalize.
    transform = transforms.Compose([
        # RandomResize(resizes=[img_size, int(1.5 * img_size), 2*img_size]),
        # transforms.RandomCrop(size=img_size),
        transforms.RandomResizedCrop(size=img_size, scale=(0.5, 1.0), ratio=(1.0, 1.0), antialias=True),

        transforms.Normalize(mean=(IMG_MEAN,), std=(IMG_STD,)),
    ])

    # Load dataset.
    dataset = PairImageDataset(data_path=data_path, transform=transform, cache_size=cache_size)

    # Split the dataset into trainig / test datasets.
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    
    # Dataloaders
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers)
    test_dataloader  = DataLoader(test_set,  batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers)

    return train_dataloader, test_dataloader
