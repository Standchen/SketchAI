import random
from functools import partial

import numpy as np
import cv2
import PIL

import torchvision.transforms as transforms

from utils import load_data


def apply_affine(line_image, sketch_image):
    """Apply same affine transformation to a pair of line & sketch images."""
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
    return tf_affine(line_image), tf_affine(sketch_image)


def overlap_two_images(a, b):
    """Overlap two images."""
    a = np.array(a, dtype=np.uint32)
    b = np.array(b, dtype=np.uint32)

    white_a = (a == 255)
    white_b = (b == 255)

    c = a + b
    c[white_a & white_b] = 255
    c[white_a ^ white_b] -= 255
    c[~(white_a & white_b)] //= 2
    return PIL.Image.fromarray(c.astype(np.uint8))


def various_thickness(img, ksize=2, iterations=1):
    """
    Adjust thickness of lines in the image.
    
    The output tends to have thicker lines as `ksize` or `iterations` increases.
    """
    if iterations == 0:
        return img

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)    
    img = cv2.bitwise_not(img)
    if iterations > 0:
        img = cv2.dilate(img, kernel=np.ones((ksize, ksize)), iterations=iterations)
    else:
        img = cv2.erode(img, kernel=np.ones((ksize, ksize)), iterations=-iterations)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    img = cv2.bitwise_not(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = PIL.Image.fromarray(img)
    return img


def overlap_data(src_data_path: str, dst_data_path:str,
                 num_overlapping: int, spawning_size: int, ksize_: list[int] = [2], iterations_: list[int] = [-1, 0, 1, 2]):
    """
    Generate overlapped image dataset. Intended exclusively for use with RAW datasets.
    
    num_overlapping: Maximum number of overlapping images.
    spawning_size: Number of images to generate for each iteration.
    ksize_: List of kernel sizes used for line thickness adjustment.
    iterations_: List of # iterations used for line thickness adjustment.
    """
    line_images, sketch_images = load_data(data_path=src_data_path)
    n = len(line_images)

    overlapped = [line_images, sketch_images]
    for ith_overlap in range(1, num_overlapping):
        line_over, sketch_over = [], []
        for i in range(spawning_size):
            prev_idx = random.randint(0, len(overlapped[0]) - 1)
            curr_idx = random.randint(0, n-1)
            line_prev, sketch_prev = overlapped[0][prev_idx], overlapped[1][prev_idx]
            line_curr, sketch_curr = line_images[curr_idx], sketch_images[curr_idx]

            # line_prev, sketch_prev = apply_affine(line_prev, sketch_prev)
            line_curr, sketch_curr = apply_affine(line_curr, sketch_curr)

            sketch_new = overlap_two_images(sketch_prev, sketch_curr)

            # Adjust thickness and save the pairs.
            for ksize in ksize_:
                for iterations in iterations_:
                    line_thick = various_thickness(img=line_curr, ksize=ksize, iterations=iterations)
                    line_new = overlap_two_images(line_prev, line_thick)

                    file_id = f"{ith_overlap}th_{i:03d}_{ksize}_{iterations}"
                    line_new.save(f"{dst_data_path}/line/{file_id}.png")
                    sketch_new.save(f"{dst_data_path}/sketch/{file_id}.png")

                    line_over.append(line_new)
                    sketch_over.append(sketch_new)

        overlapped = [line_over, sketch_over]

    # Also save non-overlapped data.
    for i, (line, sketch) in enumerate(zip(line_images, sketch_images)):
        for ksize in ksize_:
            for iterations in iterations_:
                line_thick = various_thickness(img=line, ksize=ksize, iterations=iterations)

                file_id = f"0th_{i:03d}_{ksize}_{iterations}"
                line_thick.save(f"{dst_data_path}/line/{file_id}.png")
                sketch.save(f"{dst_data_path}/sketch/{file_id}.png") 