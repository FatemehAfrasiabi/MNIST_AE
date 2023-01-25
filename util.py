import torchvision
from torchvision import transforms
from torch.utils.data import random_split
from sklearn.manifold import TSNE
import plotly.express as px
from datetime import datetime
import pandas as pd
import os
import argparse

class Util:
    def __init__(self) -> None:
        self.train_dataset = torchvision.datasets.MNIST('dataset', train=True, download=True)
        self.test_dataset = torchvision.datasets.MNIST('dataset', train=False, download=True)

        train_transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(15),
            #transforms.RandomRotation([90, 180]),
            #transforms.Resize([32, 32]),
            #transforms.RandomCrop([28, 28]),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.train_dataset.transform = train_transform
        self.test_dataset.transform = test_transform
        #length = len(self.train_dataset)
        #self.train_data, self.val_data = random_split(self.train_dataset, [int(length-length*0.2), int(length*0.2)])

    def get_sep_indx_data(self, digit_filter, train=True):
        """Creates a dataset with only one specific class
        Args:
            digit_filter (int): class number
            train (bool): return train dataset
        Returns:
            MNIST dataset: MNIST dataset with only one class
        """
        if train:
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset
        sep_indices =  dataset.targets == digit_filter
        dataset.targets = dataset.targets[sep_indices]
        dataset.data = dataset.data[sep_indices]

        return dataset

    def create_tsne_plot(self, latent_df):
        latent_df = latent_df.sort_values(
        by='target', 
        ascending=True)
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(latent_df.drop(['target'],axis=1))
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
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_traces(marker=dict(size=4,
                                    line=dict(width=0.3,
                                                color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
        fig.update_layout({ 'plot_bgcolor': 'rgba(256, 256, 256, 1)', 'paper_bgcolor': 'rgba(256, 256, 256, 1)', })
        fig.update_layout(legend_traceorder="reversed")
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        fig.write_image(f"AECompare/tsne_plots/{current_time}.png")
        fig.show()
        return fig
    
@staticmethod
def summarize_chimera_results():
    """ Read Chimera jobs results
    """
    directory = 'chimera_jobs/'
    cols = ['model', 'latent_len', 'random_seed', 'avg_accuracy']
    AE_df = pd.DataFrame(columns=cols)
    VAE_df = pd.DataFrame(columns=cols)
    for f in os.listdir(directory):
        if f.endswith(".out"): 
            file = open(directory + f, 'r')
            lines = file.readlines()
            if lines[1].startswith('VAE'):
                for i, line in enumerate(lines):
                    if i==0 or line.startswith('end'):
                        continue
                    VAE_df.loc[len(VAE_df)] = line.split()
            elif lines[1].startswith('AE'):
                for i, line in enumerate(lines):
                    if i==0 or line.startswith('end'):
                        continue
                    AE_df.loc[len(AE_df)] = line.split()
            else:
                continue
    sum_vae = VAE_df[['latent_len', 'avg_accuracy']].astype(float).groupby(VAE_df.latent_len).mean()
    sum_ae = AE_df[['latent_len', 'avg_accuracy']].astype(float).groupby(AE_df.latent_len).mean()
    sum_vae.to_csv(directory + 'summarized_results/vae_results.csv', index=False)
    sum_ae.to_csv(directory + 'summarized_results/ae_results.csv', index=False)

if __name__=="__main__":
    
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--function', choices=['SUM_RESULTS', 'PLOT_TSNE'], default='SUM_RESULTS')
    args = args_parser.parse_args()

    if args.function == 'SUM_RESULTS':
        summarize_chimera_results()
    
    