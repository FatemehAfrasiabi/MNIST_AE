import matplotlib.pyplot as plt
import numpy as np
import os
import random
import csv

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class AutoEncoder(nn.Module):
    def __init__(self, latent_len, digit, random_seed=42):
        super(AutoEncoder, self).__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = torch.nn.MSELoss()
        self.digit = digit
        self.writer = SummaryWriter()
        self.latent_len = latent_len
        self.random_seed = random_seed
        self.set_seed(random_seed)

        #print("Model info:")
        #print(
        #    f"digit:{self.digit}, device:{self.device}, latent length:{self.latent_len}, random_seed:{self.random_seed}")

        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(1, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, 400),
            # 400
            nn.ReLU(True),
            # 128
            nn.Linear(400, 128),
            nn.ReLU(True),
            # 2
            nn.Linear(128, latent_len),
            nn.Softmax(dim=1)
        )

        self.decoder = nn.Sequential(
            # 2
            nn.Linear(latent_len, 128),
            nn.ReLU(True),
            # 400
            nn.Linear(128, 400),
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            nn.ReLU(True),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

    ####################### Training the model ##########################
    def train_loop(self, dataloader, optimizer):
        self.train()
        train_loss = []
        for data, _ in dataloader:
            data = data.to(self.device)
            reconstruction = self.forward(data)
            loss = self.criterion(reconstruction, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test(self, test_dataset, batch_size=32):
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        self.eval()
        test_loss = []
        latent_spaces = []
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                reconstruction = self.forward(data)
                latent_space = self.encoder(data).cpu()
                latent_spaces.append(np.array(latent_space[0]))
                loss = self.criterion(reconstruction, data)
                test_loss.append(loss.detach().cpu().numpy())
        return np.mean(test_loss), latent_spaces

    def fit(self, train_dataset, num_epochs, lr, batch_size=32, num_workers=1):
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        self.optimizer = optim.Adam(
            self.parameters(), lr=lr, weight_decay=1e-5)
        train_loss = []
        for epoch in range(num_epochs):
            train_epoch_loss = self.train_loop(
                train_loader, self.optimizer)
            train_loss.append(train_epoch_loss)
            #print(f'Epoch {epoch+1}, train loss: {train_epoch_loss:.6f}')
            #self.writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        #torch.save(self.state_dict(),
        # f'AECompare/MNIST_digits_models/AE_models/{self.digit}_{self.latent_len}_{self.random_seed}.pth')

    def evaluate(self, test_data, n=10):
        plt.figure(figsize=(14, 4))
        for i in range(n):
            ax = plt.subplot(2, n, 1 + i)
            img = test_data[i][0].unsqueeze(0).to(self.device)
            self.eval()
            with torch.no_grad():
                rec_img = self.forward(img)
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n//2:
                ax.set_title('Original images')
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n//2:
                ax.set_title('Reconstructed images')
        plt.show()

    def store_latent(self, latent_spaces, train=False):

        if train:
            file = open(
                f'AECompare/MNIST_digits_latents/AE_latents/{self.digit}_{self.latent_len}_{self.random_seed}_train.csv', 'w+', newline='')
        else:
            file = open(
                f'AECompare/MNIST_digits_latents/AE_latents/{self.digit}_{self.latent_len}_{self.random_seed}_test.csv', 'w+', newline='')
        with file:
            write = csv.writer(file)
            write.writerows(latent_spaces)

    def set_seed(self, seed) -> None:
        """_summary_

        Args:
            seed (_type_): _description_
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        #print(f"Random seed set as {seed}")
