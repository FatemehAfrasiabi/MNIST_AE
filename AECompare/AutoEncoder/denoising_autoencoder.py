import numpy as np
import random
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class DenoisingAutoEncoder(nn.Module):
    def __init__(self, latent_len, digit, random_seed=42):
        super().__init__()
        self.digit = digit
        self.writer = SummaryWriter()
        self.latent_len = latent_len
        self.random_seed = random_seed
        self.set_seed()

        self.encoder = nn.Sequential(
            # 1 x 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            # 8 x 20 x 20
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            nn.Flatten(),
            # 8 x 20 x 20 = 3200
            nn.Linear(3200, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 128),
            # 128
            nn.ReLU(True),
            nn.Linear(128, latent_len),
            # latent_len,
            nn.ReLU(True),
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_len, 128),
            # latent_len
            nn.ReLU(True),
            nn.Linear(128, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 10 x 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 1 x 28 x 28
            nn.Sigmoid(),
            )
        
    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x, encoded_x
    
    def set_seed(self) -> None:
        seed = self.random_seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)