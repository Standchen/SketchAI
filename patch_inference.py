import argparse
import os

import glob
import gradio as gr
import numpy as np
import PIL

# torchvision
import torchvision.transforms as transforms

# torch
import torch

from models import GeneratorModel

from utils import IMG_MEAN, IMG_STD


def patch_synthesize(generator, img, patch_size: int, model_size: int):
    print(f"[*] Got {img.size = }, {patch_size = }, {model_size = }.")

    W, H = img.size
    img = np.array(img)

    generator.eval()

    # Crop patches and perform inference on them.
    res = []
    for x1 in range(0, H, patch_size):
        patches = []
        for y1 in range(0, W, patch_size):
            # Crop.
            x2 = min(H, x1 + patch_size)
            y2 = min(W, y1 + patch_size)

            ratio = model_size / patch_size
            patch_h, patch_w = x2 - x1, y2 - y1
            model_h, model_w = int(ratio * patch_h), int(ratio * patch_w)

            patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
            patch[:patch_h, :patch_w] = img[x1:x2, y1:y2]
            patch = PIL.Image.fromarray(patch)

            # Preprocessing
            transform = transforms.Compose([
                transforms.Resize(size=2*model_size, antialias=True),
                transforms.Resize(size=model_size, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=(IMG_MEAN,), std=(IMG_STD,)),
            ])

            patch = transform(patch)
            patch = patch.unsqueeze(0).to("cuda")

            # Inference
            generated_img = generator(patch)
            print(f"[!] {generated_img.shape = }")
            generated_img = generated_img[:, :, :model_h, :model_w]
            patches.append(transforms.functional.to_pil_image(torch.clamp(IMG_STD*generated_img.data.squeeze(0) + IMG_MEAN, min=0.0, max=1.0)))
        res.append(patches)

    # Calculate output size.
    out_W = 0
    for j in range(len(res[0])):
        out_W += res[0][j].size[0]

    out_H = 0
    for i in range(len(res)):
        out_H += res[i][0].size[1]

    syn_img = np.zeros((out_H, out_W), dtype=np.uint8)
    print(f"[!] {out_H = }, {out_W = }")
    for i in range(len(res)):
        for j in range(len(res[0])):
            curr_W, curr_H = res[i][j].size
            print(f"[!] {curr_W = }, {curr_H = }")
            syn_img[i*model_size:i*model_size + curr_H, j*model_size:j*model_size + curr_W] = np.array(res[i][j])

    syn_img = PIL.Image.fromarray(syn_img).convert('L')
    return syn_img


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument("--patch-size", type=int, default=384)
    parser.add_argument("--model-size", type=int, default=384)

    parser.add_argument("--input-nc", type=int, default=1)
    parser.add_argument("--output-nc", type=int, default=1)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--num-residual-blk", type=int, default=1)

    parser.add_argument("--model-path", type=str, default="./models/")

    args = parser.parse_args()

    # Set gradient calculation off.
    torch.set_grad_enabled(False)

    # Load generator model.
    generator = GeneratorModel(input_nc=args.input_nc,
                               output_nc=args.output_nc,
                               ngf=args.ngf,
                               num_residual_blk=args.num_residual_blk).to("cuda")

    # If directory is given, load the latest checkpoint.
    # Otherwise, load the specified checkpoint.
    if os.path.isdir(args.model_path):
        filenames = glob.glob(os.path.join(args.model_path, f"G_*.pt"))
        filenames.sort()
        loading_path = os.path.join(filenames[-1])
    else:
        loading_path = args.model_path
    generator.load_state_dict(torch.load(loading_path))
    print(f"[*] Loading model at {loading_path}.")

    # Launch Gradio app.
    demo = gr.Interface(
        fn=lambda img, patch_size, model_size: patch_synthesize(generator=generator, img=img, patch_size=patch_size, model_size=model_size),
        inputs=[
            # Input image
            gr.Image(type="pil", image_mode="L"),
            gr.Number(value=args.patch_size, precision=0, label="Patch Size"),
            gr.Number(value=args.model_size, precision=0, label="Model Size"),
        ],
        outputs=[
            "image"
        ],
        # live=True
    )

    demo.launch()
