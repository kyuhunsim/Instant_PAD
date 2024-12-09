import os
import torch
import torch.nn as nn
import numpy as np
from ngp.gridencoder import GridEncoder
from ngp.ffmlp import FMLP


def load_nerf(args, device):
    """Instantiate NeRF's MLP model using torch-ngp."""
    # Initialize Grid Encoder
    grid_encoder = GridEncoder(input_dim=3, level=16, per_level_scale=2.0).to(device)
    input_ch = grid_encoder.output_dim
    output_ch = 4

    # Initialize NeRF Model
    model = FMLP(input_dim=input_ch, hidden_dim=256, output_dim=output_ch).to(device)

    # Load checkpoint
    ckpt_dir = args.ckpt_dir
    ckpt_name = args.ckpt_name
    ckpt_path = os.path.join(ckpt_dir, 'LEGO-3D', f"{ckpt_name}.tar")
    print('Found ckpts:', ckpt_path)
    print('Reloading from:', ckpt_path)
    ckpt = torch.load(ckpt_path)

    # Load model state
    model.load_state_dict(ckpt["model_state_dict"])

    # Disable gradient updates
    for param in model.parameters():
        param.requires_grad = False

    render_kwargs = {
        'model': model,
        'grid_encoder': grid_encoder,
        'chunk': args.chunk,
        'perturb': args.perturb,
        'N_samples': args.N_samples,
        'N_importance': args.N_importance,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    return render_kwargs


# Rendering logic with torch-ngp
def render_rays(ray_batch, model, grid_encoder, chunk=1024 * 32):
    """Render rays using torch-ngp's optimized structure."""
    # Encode coordinates using Grid Encoder
    coords_encoded = grid_encoder(ray_batch)

    # Render rays with the model
    rgb, disp, acc, extras = render_rays(model, coords_encoded, grid_encoder, chunk=chunk)

    return rgb, disp, acc, extras


# Main NeRF training and testing pipeline
def main():
    class Args:
        def __init__(self):
            self.ckpt_dir = './checkpoints'
            self.ckpt_name = 'model'
            self.chunk = 1024 * 32
            self.perturb = 0.0
            self.N_samples = 64
            self.N_importance = 128
            self.white_bkgd = False
            self.raw_noise_std = 0.0

    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load NeRF model
    render_kwargs = load_nerf(args, device)

    # Example rendering parameters
    H, W, focal = 800, 800, 555.0
    ray_batch = torch.rand((H * W, 3)).to(device)  # Example rays

    # Render scene
    rgb, disp, acc, extras = render_rays(
        ray_batch,
        render_kwargs["model"],
        render_kwargs["grid_encoder"],
        chunk=render_kwargs["chunk"]
    )

    print("Rendering complete. RGB shape:", rgb.shape)

if __name__ == "__main__":
    main()
