"""Trains different AutoEncoder models

Author: 
    Fatemeh Afrasiabi - 01.18.2023

"""
import argparse
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import TSNE
#import plotly.express as px

from util import *
import torch.nn as nn
import AECompare.AutoEncoder.autoencoder as ae
import AECompare.AutoEncoder.variational_autoencoder as vae
import AECompare.AutoEncoder.protein_autoencoder as p_vae
import AECompare.AutoEncoder.denoising_autoencoder as dae


def train_store_latent(AE_type, epochs, latent_len, batch_size,
    learning_rate, random_seed, verbose=0):
    """Trains Individual AutoEncoders for each digit using the same parameters

    Args:
        AE_type (str): AE type ('VAE'/'AE'/etc)
        epochs (int): num of epochs
        latent_len (int): latent vector length
        batch_size (int): train batch size
        learning_rate (float): AE learning rate
        random_seed (int): random seed to be used in all AEs
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    model_dir = f'AECompare/MNIST_digits_models/{AE_type}_models'
    latent_dir = f'AECompare/MNIST_digits_latents/{AE_type}_latents'
    add_noise = False
    
    for digit in range(10):
        train_data = get_sep_indx_data(digit_filter=digit, train=True)
        if AE_type == 'AE':
            model = ae.AutoEncoder(latent_len=latent_len,
                                   digit=digit, random_seed=random_seed)
        elif AE_type == 'VAE':
            model = vae.VariationalAutoEncoder(
                latent_len=latent_len, digit=digit, random_seed=random_seed)
        elif AE_type == 'DAE':
            model = dae.DenoisingAutoEncoder(
                latent_len=latent_len, digit=digit, random_seed=random_seed)
            add_noise = True
        else:
            model = p_vae.ProtAutoEncoder(
                latent_len=latent_len, digit=digit, random_seed=random_seed)

        model.to(device)

        # Train and fit the model
        fit_model(model, train_data, epochs, learning_rate, device,
                criterion, model_dir, batch_size, verbose,
                num_workers=8, add_noise=add_noise, validation=False)
        # model.load_state_dict(torch.load(f'{model_dir}/{digit}_{latent_len}_{random_seed}.pth'))

        # Store train and test latent spaces
        train_loss, train_latent = test_model(model, train_data, device, criterion, add_noise)
        store_latent(model, train_latent, latent_dir, train=True)

        test_data = get_sep_indx_data(digit_filter=digit, train=False)
        test_loss, latent = test_model(model, test_data, device, criterion, add_noise)
        store_latent(model, latent, latent_dir, train=False)
        if verbose == 1:
            print(f"Digit {digit} train loss: {train_loss}, test loss: {test_loss}")

    return latent_dir

def process_latents(latent_dir, latent_len, random_seed):

    latent_df = pd.DataFrame()
    test_latent_df = pd.DataFrame()
    for digit in range(10):
        df = pd.read_csv(
            latent_dir + f'{digit}_{latent_len}_{random_seed}_train.csv', header=None)
        df_test = pd.read_csv(
            latent_dir + f'{digit}_{latent_len}_{random_seed}_test.csv', header=None)
        df['target'] = int(digit)
        df_test['target'] = int(digit)
        latent_df = pd.concat([latent_df, df])
        test_latent_df = pd.concat([test_latent_df, df_test])

    # Shuffle the new MNIST data
    latent_df = latent_df.sample(frac=1).reset_index(drop=True)
    test_latent_df = test_latent_df.sample(frac=1).reset_index(drop=True)
    x_train, y_train = latent_df[[a for a in range(
        0, latent_len)]].to_numpy(), latent_df['target'].to_numpy()
    x_test, y_test = test_latent_df[[a for a in range(
        0, latent_len)]].to_numpy(), test_latent_df['target'].to_numpy()
    return x_train, x_test, y_train, y_test


def classify_using_latents(x_train, x_test, y_train, y_test):
    """Classifies test data using classifier fitted on train latent spaces
        Used GNB classifier here but can change to more advanced ones

    Returns:
        float: Average accuracy score
    """
    gnb_classifier = GaussianNB().fit(x_train, y_train)
    accuracy = gnb_classifier.score(x_test, y_test)
    return accuracy


def plot_tsne(x_train, y_train):

    tsne = TSNE(n_components=2, random_state=1)
    tsne_results = tsne.fit_transform(x_train)
    fig = px.scatter(tsne_results, x=0, y=1,
                     color=y_train.astype(str),
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
    fig.show()


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--model', choices=['AE', 'VAE', 'Prot_VAE', 'DAE'], default='VAE')
    args_parser.add_argument('--epochs', type=int, default=30)
    args_parser.add_argument('--latent_len', type=int, default=25)
    args_parser.add_argument('--batch_size', type=int, default=64)
    args_parser.add_argument('--learning_rate', type=float, default=0.001)
    args_parser.add_argument('--random_seed', type=int, default=42)
    args_parser.add_argument('--verbose', choices=[0,1], default=0)
    args = args_parser.parse_args()

    # Train the model and store the latents
    latent_dir = train_store_latent(args.model, args.epochs, args.latent_len,
                                    args.batch_size, args.learning_rate,
                                    args.random_seed, args.verbose)

    # Get and process the latent spaces
    x_train, x_test, y_train, y_test = process_latents(
        latent_dir, args.latent_len, args.random_seed)

    # Mean accuracy of classification using latent spaces on Naive Bayes Gaussian Classifier
    acc_score = classify_using_latents(x_train, x_test, y_train, y_test)

    print(args.model, args.latent_len, args.random_seed, acc_score)
    #plot_tsne(x_train, y_train)
