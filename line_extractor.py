import cv2
import gradio as gr
import numpy as np


def get_line_image(img, R: int,
                   gaussian_ksize: int, gaussian_sigma: int,
                   laplacian_ksize: int,
                   thresh: int, maxval: int,
                   close_ksize: int, open_ksize: int,
                   dilate_ksize: int, dilate_iter: int,
                   area_thresh: int,
                   post_close_ksize: int,
                   erode_ksize: int, erode_iter: int,
                   median_ksize: int):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (img.shape[1]*R, img.shape[0]*R), interpolation=cv2.INTER_CUBIC)

    img = cv2.GaussianBlur(img,
                           ksize=(gaussian_ksize, gaussian_ksize),
                           sigmaX=gaussian_sigma)

    img = cv2.Laplacian(img, -1, ksize=laplacian_ksize)

    img = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)[1]

    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)))

    img = cv2.dilate(img,
                     kernel=np.ones((dilate_ksize, dilate_ksize)),
                     iterations=dilate_iter)

    res = np.zeros(img.shape, np.uint8)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8)
    for i, stat in enumerate(stats):
        if i == 0:
            continue
        area = stat[cv2.CC_STAT_AREA]
        if area >= area_thresh:
            res[labels == i] = 255
    img = res

    # Additional morphological operation
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (post_close_ksize, post_close_ksize)))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    img = cv2.erode(img, kernel=np.ones((erode_ksize, erode_ksize)), iterations=erode_iter)

    img = cv2.medianBlur(img, median_ksize)

    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img



demo = gr.Interface(
    fn=get_line_image,
    inputs=[
        # Input image
        gr.Image(),

        # Resizing
        gr.Slider(1, 4, step=1, value=2, label="R"),

        # Gaussian blur
        gr.Slider(1, 25, step=2, value=7, label="GaussianBlur ksize"),
        gr.Slider(0, 20, step=1, value=0, label="GaussianBlur sigma"),

        # Laplacian
        gr.Slider(1, 25, step=2, value=5, label="Laplacian ksize"),

        # Thresholding
        gr.Slider(0, 255, step=1, value=50, label="Threshold thresh"),
        gr.Slider(0, 255, step=1, value=255, label="Threshold maxval"),

        # Morphological transformations
        gr.Slider(0, 25, step=1, value=2, label="Morph close ksize"),
        gr.Slider(0, 25, step=1, value=3, label="Morph open ksize"),

        # Dilation
        gr.Slider(0, 25, step=1, value=2, label="Dilation ksize"),
        gr.Slider(0, 8, step=1, value=1, label="Dilation iterations"),

        # Connected Components area thresholding
        gr.Slider(0, 1000, step=1, value=200, label="Connected Components area threshold"),

        # Post morphological transformations
        gr.Slider(0, 25, step=1, value=2, label="Post morph close ksize"),

        # Erosion
        gr.Slider(0, 25, step=1, value=2, label="Erosion ksize"),
        gr.Slider(0, 8, step=1, value=1, label="Erosion iterations"),

        # Median blur
        gr.Slider(0, 25, step=2, value=5, label="MedianBlur ksize"),
    ],
    outputs=[
        "image"
    ],
    live=True
)

"""
# Resizing
R = gr.Slider(1, 4)

# Gaussian blur
gaussian_ksize = gr.Slider(1, 4)
gaussian_sigma = gr.Slider(1, 4)

# Laplacian
laplacian_ksize = gr.Slider(1, 4)

# Thresholding
thresh = gr.Slider(1, 4)
maxval = gr.Slider(1, 4)

# Morphological transformations
close_ksize = gr.Slider(1, 4)
open_ksize = gr.Slider(1, 4)

# Dilation
dilate_ksize = gr.Slider(1, 4)
dilate_iter = gr.Slider(1, 4)

# Connected Components area thresholding
area_thresh = gr.Slider(1, 4)

# Post morphological transformations
post_close_ksize = gr.Slider(1, 4)

# Erosion
erode_ksize = gr.Slider(1, 4)
erode_iter = gr.Slider(1, 4)

# Median blur
median_ksize = gr.Slider(1, 4)
"""

demo.launch()