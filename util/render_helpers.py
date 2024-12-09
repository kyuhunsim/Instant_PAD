import os
import torch
import torch.nn as nn
from tqdm import tqdm
import imageio
import numpy as np
from ngp.gridencoder import GridEncoder
from ngp.ffmlp import FMLP
from ngp.raymarching import render_rays
from util.utils import to8b

# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )
    i, j = i.t(), j.t()
    dirs = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1
    )
    rays_d = torch.sum(
        dirs[..., None, :] * c2w[:3, :3], -1
    )  # Rotate ray directions
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # Translate camera origin
    return rays_o, rays_d

# Render single frame
def render(H, W, focal, model, grid_encoder, chunk=1024 * 32, c2w=None):
    rays_o, rays_d = get_rays(H, W, focal, c2w)
    ray_batch = torch.cat([rays_o, rays_d], -1).float()

    # Grid Encoding and Raymarching
    coords_encoded = grid_encoder(ray_batch)
    rgb, disp, acc, extras = render_rays(model, coords_encoded, grid_encoder, chunk=chunk)

    return rgb, disp, acc, extras

# Render a sequence of frames
def render_path(render_poses, hwf, model, grid_encoder, chunk, savedir=None):
    H, W, focal = hwf
    rgbs, disps = [], []

    for c2w in tqdm(render_poses):
        rgb, disp, acc, _ = render(H, W, focal, model, grid_encoder, chunk, c2w=c2w)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, f"{len(rgbs) - 1:03d}.png")
            imageio.imwrite(filename, rgb8)

    return np.stack(rgbs, 0), np.stack(disps, 0)

# Main function to set up and train NeRF with torch-ngp
def main():
    # Hyperparameters and setup
    H, W, focal = 800, 800, 555.0  # Example camera parameters
    render_poses = [...]  # List of camera-to-world transformation matrices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Grid Encoder
    grid_encoder = GridEncoder(input_dim=3, level=16, per_level_scale=2.0).to(device)

    # Initialize Model
    model = FMLP(input_dim=grid_encoder.output_dim, hidden_dim=256, output_dim=4).to(device)

    # Load checkpoint if available
    ckpt_path = "path_to_checkpoint.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Checkpoint loaded.")

    # Rendering parameters
    chunk = 1024 * 32
    hwf = (H, W, focal)
    savedir = "output_renders"

    # Render sequence
    rgbs, disps = render_path(render_poses, hwf, model, grid_encoder, chunk, savedir)

    # Save results
    print("Rendering complete. Results saved in:", savedir)

if __name__ == "__main__":
    main()
