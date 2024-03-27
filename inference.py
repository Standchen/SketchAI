import argparse
import os

import glob
import gradio as gr

# torchvision
import torchvision.transforms as transforms

# torch
import torch

from models import GeneratorModel

from utils import IMG_MEAN, IMG_STD


def inference(generator, img, img_size: int):
    print(f"[*] Got {img.size = }. Resize to {img_size}.")
    generator.eval()

    transform = transforms.Compose([
        transforms.Resize(size=img_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(IMG_MEAN,), std=(IMG_STD,)),
    ])

    img = transform(img).unsqueeze(0).to("cuda")
    generated_img = generator(img)
    return transforms.functional.to_pil_image(torch.clamp(IMG_STD*generated_img.data.squeeze(0) + IMG_MEAN, min=0.0, max=1.0))


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument("--img-size", type=int, default=384)

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
        # fn=lambda img: inference(generator=generator, img=img, img_size=args.img_size),
        fn=lambda img, img_size: inference(generator=generator, img=img, img_size=img_size),
        inputs=[
            # Input image
            gr.Image(type="pil", image_mode="L"),
            gr.Number(value=args.img_size, precision=0),
        ],
        outputs=[
            "image"
        ],
        # live=True
    )

    demo.launch()
