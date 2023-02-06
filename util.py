import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from sklearn.manifold import TSNE
import plotly.express as px
from datetime import datetime
import pandas as pd
import numpy as np
import os
import argparse
import csv


def set_up_dataset(validation=False):
    """Download and transform the MNIST dataset

    Returns:
        MNIST dataset: train and test sets
    """
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(15),
        #transforms.RandomRotation([90, 180]),
        #transforms.Resize([32, 32]),
        #transforms.RandomCrop([28, 28]),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.MNIST(
        'dataset', train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.MNIST(
        'dataset', train=False, transform=test_transform, download=True)
    # If need the val dataset right away
    if validation:
        length = len(train_dataset)
        train_dataset, val_dataset = random_split(
            train_dataset, [int(length-length*0.2), int(length*0.2)])
        return train_dataset, val_dataset, test_dataset

    return train_dataset, test_dataset


def get_sep_indx_data(digit_filter, train=True):
    """Creates a dataset with only one specific class
    Args:
        digit_filter (int): class number
        train (bool): return train dataset
    Returns:
        MNIST dataset: MNIST dataset with only one class
    """
    train_dataset, test_dataset = set_up_dataset()
    if train:
        dataset = train_dataset
    else:
        dataset = test_dataset
    sep_indices = dataset.targets == digit_filter
    dataset.targets = dataset.targets[sep_indices]
    dataset.data = dataset.data[sep_indices]

    return dataset


def create_tsne_plot(latent_df):
    latent_df = latent_df.sort_values(
        by='target',
        ascending=True)
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(latent_df.drop(['target'], axis=1))
    fig = px.scatter(tsne_results, x=0, y=1,
                     color=latent_df.target.astype(str),
                     labels={'0': 'tSNE component 1', '1': 'tSNE component 2'}, color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,)
    fig.update_traces(marker=dict(size=4,
                                  line=dict(width=0.5,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.update_xaxes(showline=True, linewidth=1,
                     linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1,
                     linecolor='black', mirror=True)
    fig.update_traces(marker=dict(size=4,
                                  line=dict(width=0.3,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.update_layout({'plot_bgcolor': 'rgba(256, 256, 256, 1)',
                      'paper_bgcolor': 'rgba(256, 256, 256, 1)', })
    fig.update_layout(legend_traceorder="reversed")
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    fig.write_image(f"AECompare/tsne_plots/{current_time}.png")
    fig.show()
    return fig


def summarize_chimera_results():
    """ Read Chimera jobs results
    """
    directory = 'AECompare/chimera_jobs/'
    cols = ['model', 'latent_len', 'random_seed', 'avg_accuracy']
    AE_df = pd.DataFrame(columns=cols)
    VAE_df = pd.DataFrame(columns=cols)
    for f in os.listdir(directory):
        if f.endswith(".out"):
            file = open(directory + f, 'r')
            lines = file.readlines()
            if lines[1].startswith('VAE'):
                for i, line in enumerate(lines):
                    if i == 0 or line.startswith('end'):
                        continue
                    VAE_df.loc[len(VAE_df)] = line.split()
            elif lines[1].startswith('AE'):
                for i, line in enumerate(lines):
                    if i == 0 or line.startswith('end'):
                        continue
                    AE_df.loc[len(AE_df)] = line.split()
            else:
                continue
    sum_vae = VAE_df[['latent_len', 'avg_accuracy']].astype(
        float).groupby(VAE_df.latent_len).mean().round(3)
    sum_ae = AE_df[['latent_len', 'avg_accuracy']].astype(
        float).groupby(AE_df.latent_len).mean().round(3)
    sum_vae.to_csv(
        directory + 'summarized_results/vae_results.csv', index=False)
    sum_ae.to_csv(directory + 'summarized_results/ae_results.csv', index=False)


def fit_model(model, train_dataset, num_epochs, lr, device, criterion,
              model_dir, verbose, batch_size=32, num_workers=8, add_noise=False,
              validation=False):
    if validation:
        data_len = len(train_dataset)
        val_len = int(data_len*0.2)
        train_dataset, val_dataset = random_split(
            train_dataset, [int(data_len-val_len), val_len])
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        train_epoch_loss = train_model(
            model, train_loader, optimizer, device, criterion, add_noise)
        if validation:
            val_epoch_loss, _ = test_model(
                model, val_loader, device, criterion, add_noise)
            val_loss.append(val_epoch_loss)
            if verbose:
                print(
                    f'Epoch {epoch+1}, train loss: {train_epoch_loss:.4f}, val loss: {val_epoch_loss:.4f}')
        train_loss.append(train_epoch_loss)
        if verbose:
            print(f'Epoch {epoch+1}, train loss: {train_epoch_loss:.4f}')
    # torch.save(model.state_dict(),
    # f'{model_dir}/{model.digit}_{model.latent_len}_{model.random_seed}.pth')


def train_model(model, dataloader, optimizer, device, criterion, add_noise=False):
    model.train()
    running_loss = 0.0
    for data, _ in dataloader:
        if add_noise:
            # Add random noise to the input images
            data_noisy = data + 0.5 * torch.randn(*data.shape)
            data_noisy = (torch.clamp(data_noisy, 0, 1)).to(device)
            data = data.to(device)
            recons_data, encoded_sample = model(data_noisy)
        else:
            data = data.to(device)
            recons_data, encoded_sample = model(data)
        loss = criterion(recons_data, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


def test_model(model, test_data, device, criterion, add_noise):
    dataloader = DataLoader(
        test_data,
        batch_size=1,
        num_workers=8
    )
    model.eval()
    running_loss = 0.0
    latent_spaces = []
    with torch.no_grad():
        for data, _ in dataloader:
            if add_noise:
                # Add random noise to the input images
                data_noisy = data + 0.5 * torch.randn(*data.shape)
                data_noisy = (torch.clamp(data_noisy, 0, 1)).to(device)
                data = data.to(device)
                recons_data, encoded_sample = model(data_noisy)
            else:
                data = data.to(device)
                recons_data, encoded_sample = model(data)
            latent_space = encoded_sample.cpu()
            latent_spaces.append(np.array(latent_space[0]))
            loss = criterion(recons_data, data)
            running_loss += loss.item()
            test_loss = running_loss/len(dataloader.dataset)
    return test_loss, latent_spaces


def store_latent(model, latent_spaces, directory, train=False):
    if train:
        file = open(
            f'{directory}/{model.digit}_{model.latent_len}_{model.random_seed}_train.csv', 'w+', newline='')
    else:
        file = open(
            f'{directory}/{model.digit}_{model.latent_len}_{model.random_seed}_test.csv', 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(latent_spaces)


def encode_test(model, test_dataset):
    encoded_samples = []
    for sample in test_dataset:
        img = sample[0].unsqueeze(0).to(model.device)
        label = sample[1]
        # Encode image
        model.eval()
        with torch.no_grad():
            _, encoded_img = model.forward(img)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i,
                          enc in enumerate(encoded_img)}
        encoded_sample['label'] = label
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples)
    return encoded_samples


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--function', choices=['SUM_RESULTS', 'PLOT_TSNE'], default='SUM_RESULTS')
    args = args_parser.parse_args()

    if args.function == 'SUM_RESULTS':
        summarize_chimera_results()
