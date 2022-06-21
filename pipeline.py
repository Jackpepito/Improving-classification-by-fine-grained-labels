import os
import sys
from argparse import ArgumentParser

# pytorch lighting
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint,EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

# torch
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
import torchmetrics

# General purpose
import numpy as np
import pandas as pd
from numpy import loadtxt

# custom libraries
from data.dataloader_simple import load_cifar100, load_cifar10
from models.Classifier import LitResnet
from models.MultiHead import MultiHead

# Display image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import sklearn
import skimage
import matplotlib.cm as cm

# Clustering + Reduction
import umap
import umap.plot
import hdbscan
from sklearn.metrics import silhouette_score

from torch.utils.tensorboard import SummaryWriter

class Pipeline:
    def __init__(self,  dataset='cifar10',
                        threshold=90,
                        n_aug=5,
                        strategy='fine-tuning',
                        batch_size=200,
                        max_epochs=50,
                        min_epochs=30,
                        patience=4,
                        alpha=0.2,
                        pretrained=True, #usare i pesi su Imagenet oppure no
                        use_pseudo=True
                        ):
        #Definiamo la root
        self.ROOT_DIR = os.getcwd()
        #Dataset disponibili
        self.datasets = {"cifar10": load_cifar10, "cifar100": load_cifar100}
        self.dataset = dataset
        #Soglia percentuale che separa incerti/certi
        self.threshold = threshold
        #Numero di augmentations
        self.n_aug= n_aug
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.min_epochs= min_epochs
        self.patience=patience
        self.CHECKPOINT_PATH = os.path.join(self.ROOT_DIR, "results/checkpoints_final")
        self.dir = os.path.join(self.CHECKPOINT_PATH, 'classifier', self.dataset)
        self.dir_sub = os.path.join(self.CHECKPOINT_PATH, 'classifier_sub', self.dataset)
        self.dir_finetuning = os.path.join(self.CHECKPOINT_PATH, 'fine-tuning', self.dataset)
        self.dir_multihead = os.path.join(self.CHECKPOINT_PATH, 'multihead', self.dataset)
        # Modalità di Allenamento con Pseudo label
        self.strategy = strategy
        #Numero di pseudo labels trovate
        self.n_sub = 0
        self.alpha=alpha
        self.pretrained=pretrained
        self.use_pseudo= use_pseudo

        #varie callback utili per ricavare il path del modello
        self.checkpoint_sup_callback = ModelCheckpoint(save_weights_only=True,
                                              mode="max",
                                              monitor="val_acc",
                                              dirpath=self.dir,
                                              )
        self.checkpoint_sub_callback = ModelCheckpoint(save_weights_only=True,
                                                       mode="max",
                                                       monitor="val_acc",
                                                       dirpath=self.dir_sub,
                                                       )
        self.checkpoint_supfinetuning_callback = ModelCheckpoint(save_weights_only=True,
                                                       mode="max",
                                                       monitor="val_acc",
                                                       dirpath=self.dir_finetuning,
                                                       )

        # Setting the seed
        pl.seed_everything(42)
        # Ensure that all operations are deterministic on GPU (if used) for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        print("Device:", self.device)

    def run_pipeline(self):

        if self.use_pseudo:
            print('Training initial representation step (1/3)...')
            self.classify()
            print('Running subclass discovery step (2/3)...')
            print('features_extraction...')
            self.features_extraction()
            print('Uncertain phase...')
            self.uncertain_extraction()
            print('Reduction and clustering...')
            self.get_pseudo_labels()
            print('Train with pseudo-labels step (3/3)...')

            print('Train classifier pseudo subclass...')
            self.classify(mode='sub')
            print('Train classifier fine-tuned superclass...')
            self.final_super(strategy='fine-tune') #tolto l'else
            print('Train classifier multihead superclass...')
            print('Aplha = 0.1')
            self.final_super(strategy='multi-head', alpha=0.1)
            print('Aplha = 0.9')
            self.final_super(strategy='multi-head', alpha=0.9)
        else:
            print('Train classifier true subclass...')
            self.classify(mode='sub', use_pseudo=self.use_pseudo)
            print('Train classifier fine-tuned superclass...')
            self.final_super(strategy='fine-tune', use_pseudo=self.use_pseudo)  # tolto l'else
            print('Train classifier multihead superclass...')
            print('Aplha = 0.1')
            self.final_super(strategy='multi-head', alpha=0.1, use_pseudo=self.use_pseudo)
            print('Aplha = 0.9')
            self.final_super(strategy='multi-head', alpha=0.9, use_pseudo=self.use_pseudo)
        return

    def classify(self, mode='sup',use_pseudo=True):
        if mode == 'sup':
            trainset, trainloader, valset, valloader, testset, testloader = self.datasets[self.dataset](batch_size=200)
            model = LitResnet(pretrained=self.pretrained, num_classes=trainset.num_coarse_labels)
            trainer = pl.Trainer(default_root_dir=self.dir,  # Where to save models
                                 gpus=1 if str(self.device) == "cuda:0" else 0,  # We run on a single GPU (if possible)
                                 min_epochs=self.min_epochs,
                                 max_epochs=self.max_epochs,
                                 auto_lr_find=True,
                                 callbacks=[self.checkpoint_sup_callback,
                                            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                            LearningRateMonitor("epoch"),
                                            EarlyStopping('val_loss', patience=self.patience),
                                            TQDMProgressBar(1)]  # Log learning rate every epoch
                                 )
            trainer.tune(model, trainloader, valloader)
            trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
            checkpoint = torch.load(self.checkpoint_sup_callback.best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            trainer.test(model, dataloaders=testloader)
        else:
            if use_pseudo:
                trainset, trainloader, valset, valloader, testset, testloader = self.datasets[self.dataset](batch_size=200,
                                                                                                            pseudo_labels=True)
            else:
                trainset, trainloader, valset, valloader, testset, testloader = self.datasets[self.dataset](batch_size=200)

            self.n_sub = trainset.num_sub_labels
            print("Total sublass is {}, mode pseudo is {}".format(self.n_sub, self.use_pseudo))
            model = LitResnet(pretrained=self.pretrained, num_classes=self.n_sub, mode='sub')
            trainer = pl.Trainer(default_root_dir=self.dir_sub,  # Where to save models
                                 gpus=1 if str(self.device) == "cuda:0" else 0,  # We run on a single GPU (if possible)
                                 min_epochs=self.min_epochs,
                                 max_epochs=self.max_epochs,
                                 auto_lr_find=True,
                                 callbacks=[self.checkpoint_sub_callback,
                                            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                            LearningRateMonitor("epoch"), # Log learning rate every epoch
                                            EarlyStopping('val_loss', patience=self.patience),
                                            TQDMProgressBar(1)]
                                 )
            trainer.tune(model, trainloader, valloader)
            trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
            checkpoint = torch.load(self.checkpoint_sub_callback.best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            trainer.test(model, dataloaders=valloader)
        return

    def features_extraction(self):
        trainset, trainloader = self.datasets[self.dataset](batch_size=200, not_shuffle=True)
        checkpoint = torch.load(self.checkpoint_sup_callback.best_model_path)
        model = LitResnet(num_classes=trainset.num_coarse_labels)
        model.load_state_dict(checkpoint['state_dict'])
        #attiva hook per estrarre l'ultimo layer
        model.set_activations()
        trainer = pl.Trainer(default_root_dir=self.dir,  # Where to save models
                             gpus=1 if str(self.device) == "cuda:0" else 0,  # We run on a single GPU (if possible)
                             callbacks=[TQDMProgressBar(1)]
                             )
        trainer.test(model, dataloaders=trainloader)

        # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
        activations = {name: torch.cat(outputs, 0) for name, outputs in model.activations.items()}

        os.mkdir('sub_' + self.dataset)
        np.savetxt('sub_' + self.dataset + '/activations.csv', torch.squeeze(activations['avgpool']).tolist(), delimiter=',')
        return

    def uncertain_extraction(self):
        trainset,trainloader = self.datasets[self.dataset](batch_size=1, not_shuffle=True)
        #unc=True attiva la funzione per calcolare l'incertezza
        model = LitResnet(num_classes=trainset.num_coarse_labels, unc=True)
        #numero di augmentationse e thresholg
        model.unc_configuration(n_aug=self.n_aug, threshold=self.threshold)
        checkpoint = torch.load(self.checkpoint_sup_callback.best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        trainer = pl.Trainer(default_root_dir=self.dir,  # Where to save models
                             gpus=1 if str(self.device) == "cuda:0" else 0,  # We run on a single GPU (if possible)
                             callbacks=[TQDMProgressBar(1)]
                             )
        trainer.test(model, dataloaders=trainloader)
        np.savetxt('sub_' + self.dataset + '/incertezza.csv', model.unc_values, delimiter=',')
        np.savetxt('sub_' + self.dataset + '/superclassi.csv', model.sup, delimiter=',')

    def get_pseudo_labels(self):
        features = loadtxt('sub_' + self.dataset + '\\activations.csv', delimiter=',')
        df = pd.DataFrame(features)
        df['incertezza'] = loadtxt('sub_' + self.dataset + '\incertezza.csv', delimiter=',')
        df['superclasse'] = loadtxt('sub_' + self.dataset + '\superclassi.csv', delimiter=',')
        # lista delle superclassi
        Uniquesuper = df.superclasse.unique()
        # I vari dataframe divisi per superclasse
        DataFrameDict = {elem: pd.DataFrame for elem in Uniquesuper}
        for key in DataFrameDict.keys():
            DataFrameDict[key] = df[:][df.superclasse == key]
        for key in DataFrameDict.keys():
            df = DataFrameDict[key]  # dataframe di una superclasse
            px = df.sort_values(by=['incertezza'])  # ordiniamo in ordine crescente
            first = px.head(int(len(px) * (self.threshold / 100)))
            # salviamo gli indici degli elementi più certi nel dataframe della superclasse
            indexes = first.index.tolist()
            for index in indexes:
                df.loc[index, 'incerto'] = 0  # gli elementi più certi li settiamo a 0
            df['incerto'].fillna(1, inplace=True)  # gli altri a 1, cosi abbiamo la nostra maschera
            DataFrameDict[key] = df
        num_sub = 0
        #clusterizziamo solo gli elementi meno incerti che saranno settati nella mask.csv a 0
        for key in DataFrameDict.keys():
            df_super = DataFrameDict[key].loc[DataFrameDict[key]['incerto'] == 0]
            data = df_super.drop(columns=['incerto', 'incertezza', 'superclasse'])
            mapper = umap.UMAP(
                n_neighbors=30,
                min_dist=0.0,
                n_components=4,
                random_state=42,
            ).fit(data)
            data = mapper.embedding_
            clusterer = hdbscan.HDBSCAN(min_cluster_size=100, prediction_data=True).fit(data)
            soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
            cluster_labels = [np.argmax(x)
                              for x in soft_clusters]
            #np.savetxt('sub_' + self.dataset + '/' + str(key) + '_embeddings.csv', data, delimiter=',')
            #np.savetxt('sub_' + self.dataset + '/' + str(key) + '_clusters.csv', cluster_labels, delimiter=',')
            silhouette_avg = silhouette_score(data, cluster_labels)
            writer = SummaryWriter('./run/sub_' + self.dataset + '/' + str(key) + '_embeddings_' + str(silhouette_avg) +'.csv')
            writer.add_embedding(data, metadata=cluster_labels, global_step=1)
            df_super['subclass'] = cluster_labels
            df_super['subclass'] = [x + num_sub for x in df_super['subclass']]
            num_sub += df_super['subclass'].nunique()
            indexes = df_super.index.tolist()
            for index in indexes:
                DataFrameDict[key].loc[index, 'subclass'] = df_super.loc[index, 'subclass']
            DataFrameDict[key]['subclass'].fillna(-1, inplace=True)  # gli altri a -1
        final = pd.DataFrame()
        for key in DataFrameDict.keys():
            final = pd.concat([final, DataFrameDict[key]])
        final=final.sort_index()
        np.savetxt('sub_' + self.dataset + '/pseudo.csv', final['subclass'].astype(int), delimiter=',')
        np.savetxt('sub_' + self.dataset + '/mask.csv', final['incerto'].astype(int), delimiter=',')
        np.savetxt('sub_' + self.dataset + '/new_super.csv', final['superclasse'].astype(int), delimiter=',')

    def final_super(self, strategy='fine-tune', alpha=0.5, use_pseudo=True):
        if strategy == 'fine-tune':
            # carichiamo il modello allenato sulle sottoclassi, cambiamo l'output e frezziamo i parametri di tutta la rete
            # esclusi classificatore e batch-norm
            trainset, trainloader, valset, valloader, testset, testloader = self.datasets[self.dataset](batch_size=200)
            checkpoint = torch.load(self.checkpoint_sub_callback.best_model_path)
            model = LitResnet(num_classes=self.n_sub)
            model.load_state_dict(checkpoint['state_dict'])
            model.fine_tuning(dim_out=trainset.num_coarse_labels, freeze=False)
            trainer = pl.Trainer(default_root_dir=self.dir_finetuning ,  # Where to save models
                                 gpus=1 if str(self.device) == "cuda:0" else 0,  # We run on a single GPU (if possible)
                                 min_epochs=self.min_epochs,
                                 max_epochs=self.max_epochs,
                                 auto_lr_find=True,
                                 callbacks=[self.checkpoint_supfinetuning_callback,
                                            # Save the best checkpoint based on the maximum val_acc recorded.
                                            # Saves only weights and not optimizer
                                            LearningRateMonitor("epoch"),
                                            EarlyStopping('val_loss', patience=self.patience),
                                            TQDMProgressBar(1)]  # Log learning rate every epoch
                                 )
            trainer.tune(model, trainloader, valloader)
            trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
            checkpoint = torch.load(self.checkpoint_supfinetuning_callback.best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            trainer.test(model, dataloaders=testloader)
        if strategy == 'multi-head':
            if use_pseudo:
                trainset, trainloader, valset, valloader, testset, testloader = self.datasets[self.dataset](batch_size=200,
                                                                                                            pseudo_labels=True)
            else:
                trainset, trainloader, valset, valloader, testset, testloader = self.datasets[self.dataset](batch_size=200)

            model = MultiHead(num_classes=trainset.num_sub_labels,
                              num_superclasses=trainset.num_coarse_labels,
                              pretrained=self.pretrained,
                              alpha=alpha)

            checkpoint_multihead_callback = ModelCheckpoint(
                save_weights_only=True,
                mode="max",
                monitor="val_acc sup",
                dirpath=self.dir_multihead,
            )

            trainer = pl.Trainer(default_root_dir=self.dir_multihead,  # Where to save models
                                 gpus=1 if str(self.device) == "cuda:0" else 0,  # We run on a single GPU (if possible)
                                 min_epochs=self.min_epochs,
                                 max_epochs=self.max_epochs,
                                 auto_lr_find=True,
                                 callbacks=[checkpoint_multihead_callback,
                                            # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                            LearningRateMonitor("epoch"),
                                            EarlyStopping('val_loss sup',patience=self.patience),
                                            TQDMProgressBar(1)],  # Log learning rate every epoch
                                 )
            trainer.tune(model, trainloader, valloader)
            trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
            checkpoint = torch.load(checkpoint_multihead_callback.best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            trainer.test(model, dataloaders=testloader)
        return

