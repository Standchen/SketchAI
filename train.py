import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# torch
import torch
import torch.nn.functional as F
import torch.optim as optim

from augmentation import overlap_data
from dataloader import get_dataloader
from models import GeneratorModel, DiscriminatorModel
from perceptual_loss import get_perceptual_layers
from utils import draw_images


def get_d_loss(D, x, y, target):
    """Aggregate patchGAN discriminator loss output into a scalar."""
    d_out = D(x, y)
    target = torch.full(d_out.shape, float(target)).to("cuda")
    # loss = F.mse_loss(input=d_out, target=target)
    loss = F.binary_cross_entropy_with_logits(input=d_out, target=target)
    return loss


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--do-overlapping", type=bool, default=False)

    parser.add_argument("--input-nc", type=int, default=1)
    parser.add_argument("--output-nc", type=int, default=1)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--num-residual-blk", type=int, default=1)

    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=6)

    parser.add_argument("--num-epoch", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=5e1)
    parser.add_argument("--beta", type=float, default=1e0)

    parser.add_argument("--load-model-at", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--show-every", type=int, default=1)

    parser.add_argument("--raw-data-path", type=str, default="./data/raw/")
    parser.add_argument("--overlapped-data-path", type=str, default="./data/overlapped/")
    parser.add_argument("--output-path", type=str, default="./outputs/")
    parser.add_argument("--model-path", type=str, default="./models/")

    args = parser.parse_args()

    # Assert sketch & line data to exist.
    assert os.path.isdir(f"{args.raw_data_path}/line/") and os.path.isdir(f"{args.raw_data_path}/sketch/")

    # Make directories for augmented data.
    os.makedirs(f"{args.overlapped_data_path}/line/", exist_ok=True)
    os.makedirs(f"{args.overlapped_data_path}/sketch/", exist_ok=True)

    # Make a directory for model checkpoints.
    os.makedirs(args.model_path, exist_ok=True)

    # Make a directory for outputs (intermediate results, loss plots).
    os.makedirs(args.output_path, exist_ok=True)

    # Data Augmentation
    if args.do_overlapping:
        overlap_data(src_data_path=args.raw_data_path,
                     dst_data_path=args.overlapped_data_path,
                     num_overlapping=3, spawning_size=100,
                     iterations_=[0])

    # Get train & test dataloaders.
    train_dataloader, test_dataloader = get_dataloader(data_path=args.overlapped_data_path,
                                                       batch_size=args.batch_size,
                                                       img_size=args.img_size,
                                                       cache_size=2*args.img_size,
                                                       num_workers=args.num_workers)

    # Initialize models.
    Generator = GeneratorModel(input_nc=args.input_nc, output_nc=args.output_nc, ngf=args.ngf, num_residual_blk=args.num_residual_blk).to("cuda")
    Discriminator_L = DiscriminatorModel(input_nc=args.output_nc, ndf=32, nlayers=4).to("cuda")
    Discriminator_M = DiscriminatorModel(input_nc=args.output_nc, ndf=32, nlayers=3).to("cuda")
    Discriminator_S = DiscriminatorModel(input_nc=args.output_nc, ndf=32, nlayers=2).to("cuda")

    Generator.train()
    Discriminator_L.train()
    Discriminator_M.train()
    Discriminator_S.train()

    # Initialize optimizers.
    G_optimizer = optim.Adadelta(Generator.parameters())
    D_L_optimizer = optim.Adadelta(Discriminator_L.parameters())
    D_M_optimizer = optim.Adadelta(Discriminator_M.parameters())
    D_S_optimizer = optim.Adadelta(Discriminator_S.parameters())

    # Prepare perceptual loss layers.
    vgg19, perceptual_layers = get_perceptual_layers(perceptual_layers=["relu1_1", "relu2_1", "relu3_1", "relu4_1"], pooling="max")
    vgg19.requires_grad_(False)

    # Train.
    if args.load_model_at != -1:
        Generator.load_state_dict(torch.load(f"{args.model_path}/G_{args.load_model_at}.pt"))
        Discriminator_L.load_state_dict(torch.load(f"{args.model_path}/D-L_{args.load_model_at}.pt"))
        Discriminator_M.load_state_dict(torch.load(f"{args.model_path}/D-M_{args.load_model_at}.pt"))
        Discriminator_S.load_state_dict(torch.load(f"{args.model_path}/D-S_{args.load_model_at}.pt"))

    start_epoch = args.load_model_at + 1 if args.load_model_at != -1 else 0
    g_losses, per_losses, g_fake_losses = [], [], []
    d_losses, d_real_losses, d_fake_losses = [], [], []

    for epoch in range(start_epoch, start_epoch + args.num_epoch):
        with tqdm(train_dataloader, desc=f"Epoch {epoch}") as pbar:
            it = 0
            for line_imgs, sketch_imgs in pbar:
                line_imgs = line_imgs.to("cuda")
                sketch_imgs = sketch_imgs.to("cuda")

                #######################################
                ######### Train discriminator #########
                #######################################
                generated_imgs = Generator(line_imgs)
                
                Discriminator_L.requires_grad_(True)
                Discriminator_M.requires_grad_(True)
                Discriminator_S.requires_grad_(True)

                d_real_loss_L = get_d_loss(D=Discriminator_L, x=line_imgs, y=sketch_imgs, target=True)
                d_real_loss_M = get_d_loss(D=Discriminator_M, x=line_imgs, y=sketch_imgs, target=True)
                d_real_loss_S = get_d_loss(D=Discriminator_S, x=line_imgs, y=sketch_imgs, target=True)
                d_real_loss = d_real_loss_L + d_real_loss_M + d_real_loss_S

                d_fake_loss_L = get_d_loss(D=Discriminator_L, x=line_imgs, y=generated_imgs.detach(), target=False)
                d_fake_loss_M = get_d_loss(D=Discriminator_M, x=line_imgs, y=generated_imgs.detach(), target=False)
                d_fake_loss_S = get_d_loss(D=Discriminator_S, x=line_imgs, y=generated_imgs.detach(), target=False)
                d_fake_loss = d_fake_loss_L + d_fake_loss_M + d_fake_loss_S

                d_loss_L = d_real_loss_L + d_fake_loss_L
                d_loss_M = d_real_loss_M + d_fake_loss_M
                d_loss_S = d_real_loss_S + d_fake_loss_S

                d_loss = (d_real_loss + d_fake_loss) / 2

                D_L_optimizer.zero_grad()
                d_loss_L.backward()
                D_L_optimizer.step()

                D_M_optimizer.zero_grad()
                d_loss_M.backward()
                D_M_optimizer.step()

                D_S_optimizer.zero_grad()
                d_loss_S.backward()
                D_S_optimizer.step()

                #######################################
                ########### Train generator ###########
                #######################################
                """
                Get perceptual loss
                """
                vgg19(generated_imgs)
                generated_per = [layer.x for layer in perceptual_layers]

                vgg19(line_imgs)
                sketch_per = [layer.x for layer in perceptual_layers]

                per_loss = sum([F.mse_loss(input=g, target=s) / (g.shape[0] * g.shape[1] * g.shape[2] * g.shape[3]) for g, s in zip(generated_per, sketch_per)])

                """
                Get adversarial loss
                """
                Discriminator_L.requires_grad_(False)
                Discriminator_M.requires_grad_(False)
                Discriminator_S.requires_grad_(False)

                g_fake_loss_L = get_d_loss(D=Discriminator_L, x=line_imgs, y=generated_imgs, target=True)
                g_fake_loss_M = get_d_loss(D=Discriminator_M, x=line_imgs, y=generated_imgs, target=True)
                g_fake_loss_S = get_d_loss(D=Discriminator_S, x=line_imgs, y=generated_imgs, target=True)
                g_fake_loss = g_fake_loss_L + g_fake_loss_M + g_fake_loss_S

                # Generator loss
                g_loss = (args.alpha * per_loss) + (args.beta * g_fake_loss)

                G_optimizer.zero_grad()
                g_loss.backward()
                G_optimizer.step()

                # Record & update the loss status.
                g_losses.append(g_loss.item())
                per_losses.append(per_loss.item())
                g_fake_losses.append((g_fake_loss_L.item(), g_fake_loss_M.item(), g_fake_loss_S.item()))

                d_losses.append(d_loss.item())
                d_real_losses.append((d_real_loss_L.item(), d_real_loss_M.item(), d_real_loss_S.item()))
                d_fake_losses.append((d_fake_loss_L.item(), d_fake_loss_M.item(), d_fake_loss_S.item()))

                pbar.set_postfix_str(
                    f"G: {g_loss:.3f} (per: {per_loss:.3f}, Fake: {g_fake_loss:.3f} ({g_fake_loss_L:.3f}, {g_fake_loss_M:.3f}, {g_fake_loss_S:.3f})) / D: {d_loss:.3f} (Real: {d_real_loss:.3f}, Fake: {d_fake_loss:.3f}) ({d_loss_L:.3f}, {d_loss_M:.3f}, {d_loss_S:.3f})"
                )

                it += 1

            # Show intermediate results.
            if epoch % args.show_every == 0:
                draw_images(line_imgs=line_imgs, sketch_imgs=sketch_imgs, generator=Generator,
                            n=min(args.batch_size, 5), save_path=f"{args.output_path}/intermediate_{epoch:03}.png")

            # Save model checkpoints.
            if epoch % args.save_every == 0:
                torch.save(Generator.state_dict(), f"{args.model_path}/G_{epoch}.pt")
                torch.save(Discriminator_L.state_dict(), f"{args.model_path}/D-L_{epoch}.pt")
                torch.save(Discriminator_M.state_dict(), f"{args.model_path}/D-M_{epoch}.pt")
                torch.save(Discriminator_S.state_dict(), f"{args.model_path}/D-S_{epoch}.pt")


    # Plot losses.
    it = np.arange(len(g_losses))
    plt.plot(it, g_losses, label="G")
    plt.plot(it, d_losses, label="D")
    plt.legend(loc="upper right")
    plt.title("Generator & Discriminator Loss")
    plt.savefig(f"{args.output_path}/GD_loss.png")
    plt.close()

    it = np.arange(len(g_losses))
    plt.plot(it, g_losses)
    plt.title("Generator Loss")
    plt.savefig(f"{args.output_path}/G_loss.png")
    plt.close()

    it = np.arange(len(d_losses))
    plt.plot(it, d_losses)
    plt.title("Discriminator Loss")
    plt.savefig(f"{args.output_path}/D_loss.png")
    plt.close()

    it = np.arange(len(per_losses))
    plt.plot(it, per_losses)
    plt.title("Perceptual Loss")
    plt.savefig(f"{args.output_path}/P_loss.png")
    plt.close()

    it = np.arange(len(g_fake_losses))
    plt.plot(it, list(map(lambda x: sum(x), g_fake_losses)), label="Total")
    plt.plot(it, list(map(lambda x: x[0], g_fake_losses)), label="L")
    plt.plot(it, list(map(lambda x: x[1], g_fake_losses)), label="M")
    plt.plot(it, list(map(lambda x: x[2], g_fake_losses)), label="S")
    plt.legend(loc="upper right")
    plt.title("G Fake Losses")
    plt.savefig(f"{args.output_path}/G_fake_losses.png")
    plt.close()

    it = np.arange(len(d_real_losses))
    plt.plot(it, list(map(lambda x: sum(x), d_real_losses)), label="Total")
    plt.plot(it, list(map(lambda x: x[0], d_real_losses)), label="L")
    plt.plot(it, list(map(lambda x: x[1], d_real_losses)), label="M")
    plt.plot(it, list(map(lambda x: x[2], d_real_losses)), label="S")
    plt.legend(loc="upper right")
    plt.title("D Real Losses")
    plt.savefig(f"{args.output_path}/D_real_losses.png")
    plt.close()

    it = np.arange(len(d_fake_losses))
    plt.plot(it, list(map(lambda x: sum(x), d_fake_losses)), label="Total")
    plt.plot(it, list(map(lambda x: x[0], d_fake_losses)), label="L")
    plt.plot(it, list(map(lambda x: x[1], d_fake_losses)), label="M")
    plt.plot(it, list(map(lambda x: x[2], d_fake_losses)), label="S")
    plt.legend(loc="upper right")
    plt.title("D Fake Losses")
    plt.savefig(f"{args.output_path}/D_fake_losses.png")
    plt.close()


    # Test.
    Generator.eval()

    for line_imgs, sketch_imgs in tqdm(test_dataloader):
        line_imgs = line_imgs.to("cuda")
        sketch_imgs = sketch_imgs.to("cuda")

        draw_images(line_imgs=line_imgs, sketch_imgs=sketch_imgs, generator=Generator,
                    n=min(args.batch_size, 5), save_path=f"{args.output_path}/final_output.png")
        break
