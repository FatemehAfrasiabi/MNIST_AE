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

import util as u
import AutoEncoder.autoencoder as ae
import AutoEncoder.variational_autoencoder as vae
import AutoEncoder.protein_autoencoder as p_vae

def train_store_latent(AE_type, epochs, latent_len, batch_size, learning_rate, random_seed):
    

    for digit in range(10):
        util = u.Util()
        train_data = util.get_sep_indx_data(digit_filter=digit, train=True)
        if AE_type == 'AE':
            model = ae.AutoEncoder(latent_len=latent_len, digit=digit, random_seed=random_seed)
        elif AE_type == 'VAE':
            model = vae.VariationalAutoEncoder(latent_len=latent_len, digit=digit, random_seed=random_seed)
        else:
            model = p_vae.ProtAutoEncoder(latent_len=latent_len, digit=digit, random_seed=random_seed)

        model.to(model.device)
        model.fit(train_data, num_epochs=epochs, lr=learning_rate, batch_size=batch_size)
        #VAE.load_state_dict(torch.load(f'MNIST_digits_models/VAE_models/{digit}_model_{LATENT_LENGTH}_{RANDOM_SEED}.pth'))
        #print(train_data)
        train_loss, train_latent = model.test(train_data, batch_size=1)
        model.store_latent(train_latent, train=True)
        #print(digit, train_loss, len(train_latent))
        test_data = util.get_sep_indx_data(digit_filter=digit, train=False)
        test_loss, latent = model.test(test_data, batch_size=1)
        model.store_latent(latent, train=False)
        #print(digit, test_loss, len(latent))

def process_latents(AE_type, latent_len, random_seed):
    if AE_type == 'AE':
        latent_path = 'MNIST_digits_latents/AE_latents/'
    elif AE_type == 'VAE':
        latent_path = 'MNIST_digits_latents/VAE_latents/'
    else: 
        latent_path = 'MNIST_digits_latents/Prot_AE_latents/'

    latent_df = pd.DataFrame()
    test_latent_df = pd.DataFrame()
    for digit in range(10):
        df = pd.read_csv(latent_path + f'{digit}_{latent_len}_{random_seed}_train.csv', header=None)
        df_test = pd.read_csv(latent_path + f'{digit}_{latent_len}_{random_seed}_test.csv', header=None)
        df['target'] = int(digit)
        df_test['target'] = int(digit)
        latent_df = pd.concat([latent_df, df])
        test_latent_df = pd.concat([test_latent_df, df_test])

    # Shuffle the new MNIST data    
    latent_df = latent_df.sample(frac=1).reset_index(drop=True)
    test_latent_df = test_latent_df.sample(frac=1).reset_index(drop=True)
    x_train, y_train = latent_df[[a for a in range(0,latent_len)]].to_numpy(), latent_df['target'].to_numpy()
    x_test, y_test = test_latent_df[[a for a in range(0,latent_len)]].to_numpy(), test_latent_df['target'].to_numpy()
    return x_train, x_test, y_train, y_test

def classify_using_latents(x_train, x_test, y_train, y_test):
    gnb_classifier = GaussianNB().fit(x_train, y_train)
    accuracy = gnb_classifier.score(x_test, y_test)
    #print("Accuracy:", accuracy)
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
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_traces(marker=dict(size=4,
                                line=dict(width=0.3,
                                            color='DarkSlateGrey')),
                    selector=dict(mode='markers'))
    fig.update_layout({ 'plot_bgcolor': 'rgba(256, 256, 256, 1)', 'paper_bgcolor': 'rgba(256, 256, 256, 1)', })
    fig.update_layout(legend_traceorder="reversed")
    fig.show()


if __name__=="__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--model', choices=['AE', 'VAE', 'Prot_VAE'], default='AE')
    args_parser.add_argument('--epochs', type=int, default=30)
    args_parser.add_argument('--latent_len', type=int, default=25)
    args_parser.add_argument('--batch_size', type=int, default=64)
    args_parser.add_argument('--learning_rate', type=float, default=0.001)
    args_parser.add_argument('--random_seed', type=int, default=42)
    args = args_parser.parse_args()
    # Train the model and store the latents
    train_store_latent(args.model, args.epochs, args.latent_len, args.batch_size, args.learning_rate, args.random_seed)

    # Get and process the latent spaces 
    x_train, x_test, y_train, y_test = process_latents(args.model, args.latent_len, args.random_seed)

    # Mean accuracy of classification using latent spaces on Naive Bayes Gaussian Classifier
    acc_score = classify_using_latents(x_train, x_test, y_train, y_test)

    print(args.model, args.latent_len, args.random_seed, acc_score)
    #plot_tsne(x_train, y_train)
