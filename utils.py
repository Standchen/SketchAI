import glob
import os

import matplotlib.pyplot as plt
import PIL

import torchvision.transforms as transforms
import torch


IMG_MEAN = 0.98
IMG_STD = 0.11
# IMG_MEAN = 0.0
# IMG_STD = 1.0

def load_data(data_path, colorspace="L", size=None, get_filenames=False):
    """
    Load image data from given path.
    
    While loading images, convert colorspace or size if specified.
    """
    line_filenames   = glob.glob(f"{data_path}/line/*")
    sketch_filenames = glob.glob(f"{data_path}/sketch/*")

    # Check file structure consistency.
    line_filenames.sort()
    sketch_filenames.sort()
    line_fns = [os.path.basename(filename) for filename in line_filenames]
    sketch_fns = [os.path.basename(filename) for filename in sketch_filenames]
    assert line_fns == sketch_fns, "Inconsistent image file structures"

    # Load images.
    # If size is specified, resize images accordingly.
    if size is not None:
        line_images = [PIL.Image.open(filename).resize((size, size)).convert(colorspace) for filename in line_filenames]
        sketch_images = [PIL.Image.open(filename).resize((size, size)).convert(colorspace) for filename in sketch_filenames]
    else:
        line_images = [PIL.Image.open(filename).convert(colorspace) for filename in line_filenames]
        sketch_images = [PIL.Image.open(filename).convert(colorspace) for filename in sketch_filenames]

    # Check size consistency.
    sz = line_images[0].size
    for i, img in enumerate(line_images):
        assert img.size == sz, f"Size inconsistency at {line_filenames[i]} (Expected {sz}, but {img.size} found)"
    for i, img in enumerate(sketch_images):
        assert img.size == sz, f"Size inconsistency at {line_filenames[i]} (Expected {sz}, but {img.size} found)"

    if get_filenames:
        return line_images, sketch_images, line_fns
    else:
        return line_images, sketch_images


def draw_images(line_imgs, sketch_imgs, generator, n, save_path):
    """
    Draw line, sketch, and generated images for comparison.
    """
    generator.eval()

    with torch.no_grad():
        generated_imgs = generator(line_imgs[:n].to("cuda"))

        # plt.figure(figsize=(60, 20*n + 10))
        plt.figure(figsize=(30, 10*n + 5))
        for i in range(n):
            plt.subplot(n, 3, 3*i + 1)
            plt.imshow(transforms.functional.to_pil_image(torch.clamp(IMG_STD*line_imgs[i].data.squeeze(0) + IMG_MEAN, min=0.0, max=1.0)), cmap="gray")
            plt.title("Line Image")

            plt.subplot(n, 3, 3*i + 2)
            plt.imshow(transforms.functional.to_pil_image(torch.clamp(IMG_STD*generated_imgs[i].data.squeeze(0) + IMG_MEAN, min=0.0, max=1.0)), cmap="gray")
            plt.title("Generated Image")

            plt.subplot(n, 3, 3*i + 3)
            plt.imshow(transforms.functional.to_pil_image(torch.clamp(IMG_STD*sketch_imgs[i].data.squeeze(0) + IMG_MEAN, min=0.0, max=1.0)), cmap="gray")
            plt.title("Sketch Image")

        plt.savefig(save_path)
        plt.close()

    generator.train()
