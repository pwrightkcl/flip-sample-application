import torch.nn as nn

from diffusion_model_unet import DiffusionModelUNet


class SimpleNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=3,
            num_res_blocks=1,
            num_channels=[32, 64, 64],
            attention_levels=(False, True, True),
            num_head_channels=1,
        )