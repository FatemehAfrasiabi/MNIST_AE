import matplotlib.pyplot as plt
import numpy as np 
import torch
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import random 

class ProtAutoEncoder(nn.Module):
        def __init__(self, latent_len, digit, random_seed=42):
            super(ProtAutoEncoder, self).__init__()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.digit = digit
            self.writer = SummaryWriter()
            self.latent_len = latent_len
            self.random_seed = random_seed
            self.set_seed()

            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                nn.ReLU(),
                )

            self.z_mean = nn.Linear(128, latent_len)
            self.z_log_sigma = nn.Linear(128, latent_len)

            self.decoder = nn.Sequential(
                # 2
                nn.Linear(latent_len, 128),
                nn.ReLU(),

                nn.Linear(128, 784),
                nn.ReLU(),

                nn.Unflatten(1, (1, 28, 28)),
                nn.Sigmoid(),
                )
            self.N = torch.distributions.Normal(0, 1)
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
        def sampling(self, mu, sigma):
            #epsilon = torch.empty(mu.size()).normal_(mean=0., std=0.1)
            #print(f'epsilon device {epsilon.to(self.device).get_device()}, mu {mu.get_device()}, sigma {sigma.get_device()}')
            #s1 = mu + torch.exp(sigma) * epsilon.to(self.device)

            # std = torch.exp(0.5*sigma)
            # eps = torch.randn_like(std)
            # s2 =  eps.mul(std).add_(mu)
            sample = mu + torch.exp(sigma) * self.N.sample(mu.shape)

            return sample

        def forward(self, x):
            x = self.encoder(x)
            mu = self.z_mean(x)

            sigma = self.z_log_sigma(x)
            z = self.sampling(mu, sigma)

            #self.encoded_sample = z
            dec = self.decoder(z)
            return dec, mu, sigma, z

        def loss_function(self, recon_x, x, mu, sigma):
            #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114

            # Correct KLD
            #KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

            sigma = torch.exp(sigma)
            KLD2 = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            loss2 = ((x - recon_x)**2).sum() + KLD2
            return  loss2 #BCE + KLD

        ####################### Training the model ##########################
        def train_loop(self, dataloader, optimizer):
            self.train()
            running_loss = 0.0
            for data, _ in dataloader:
                data = data.to(self.device)
                reconstruction, mu, sigma, _ = self.forward(data)
                loss = self.loss_function(reconstruction, data, mu, sigma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss/len(dataloader.dataset)
            return train_loss

        def test(self, test_loader):
            self.eval()
            running_loss = 0.0
            latent_spaces = []
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(self.device)
                    reconstruction , mu, sigma, encoded_sample = self.forward(data)
                    
                    latent_space = encoded_sample.cpu()
                    latent_spaces.append(np.array(latent_space[0]))

                    loss = self.loss_function(reconstruction, data, mu, sigma)
                    running_loss += loss.item()
            val_loss = running_loss/len(test_loader.dataset)
            return val_loss, latent_spaces

        def fit(self, train_dataset, num_epochs, lr, batch_size=32):
            m = len(train_dataset)
            val_len = int(m*0.2)
            train_data, val_data = random_split(train_dataset, [int(m-val_len), val_len])
            train_loader = DataLoader(
                train_data,
                batch_size=batch_size,
                num_workers=32
            )
            val_loader = DataLoader(
                val_data,
                batch_size=batch_size,
                num_workers=32
            )
            self.lr = lr
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
            train_loss = []
            for epoch in range(num_epochs):
                train_epoch_loss = self.train_loop(train_loader, optimizer)
                val_epoch_loss, _ = self.test(val_loader)
                train_loss.append(train_epoch_loss)
                #print(f'Epoch {epoch+1}, train loss: {train_epoch_loss:.4f}, val loss: {val_epoch_loss:.4f}')
                self.writer.add_scalar('Loss/train', train_epoch_loss, epoch)
                self.writer.add_scalar('Loss/val', val_epoch_loss, epoch)
            torch.save(self.state_dict(), f'AECompare/MNIST_digits_models/Prot_models/{self.digit}_{self.lr}_{self.latent_len}_{self.random_seed}.pth')

        def evaluate(self, test_data, n=10):
            plt.figure(figsize=(16,4.5))
            for i in range(n):
                ax = plt.subplot(2,n, 1+ i)
                img = test_data[i][0].unsqueeze(0).to(self.device)
                self.eval()
                with torch.no_grad():
                    rec_img, mu, sigma, encoded_sample  = self.forward(img)
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
            import csv
            if train:
                file = open(f'AECompare/MNIST_digits_latents/Prot_AE_latents/{self.digit}_{self.lr}_{self.latent_len}_{self.random_seed}_train.csv', 'w+', newline='')
            else:
                file = open(f'AECompare/MNIST_digits_latents/Prot_AE_latents/{self.digit}_{self.lr}_{self.latent_len}_{self.random_seed}_test.csv', 'w+', newline='')
            with file:
                write = csv.writer(file)
                write.writerows(latent_spaces)

        def set_seed(self) -> None:
            seed = self.random_seed
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



