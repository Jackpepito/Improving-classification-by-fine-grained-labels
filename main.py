import os
import sys
from argparse import ArgumentParser

# pytorch lighting
import pytorch_lightning as pl
# torch
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torchvision.models.resnet import resnet18

# custom libraries
from data.dataloader import load_cifar100, load_cifar10, load_mnist
from models.BaseModel import BaseModel
from models.MultiHeadModel import MultiHead
from strategies import BaseStrategy, MultiHeadStrategy

conf_path = os.getcwd() + "/new experiments/"
sys.path.append(conf_path)
sys.path.append(conf_path + '/data')
sys.path.append(conf_path + '/strategies')
sys.path.append(conf_path + '/models')
sys.path.append(conf_path + '/utils')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

datasets = {"mnist": load_mnist, "cifar10": load_cifar10, "cifar100": load_cifar100}
# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "results/checkpoints"
CHECKPOINT_PATH=os.path.join(ROOT_DIR,CHECKPOINT_PATH)

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def main():
    lecun_fix()
    # Setting the seed
    pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    # parse parameters
    parser = ArgumentParser(description='experiment', allow_abbrev=False)
    parser.add_argument('--dataset', type=str, default="cifar10",
                        help='dataset name.', choices=["mnist", "cifar10", "cifar100"])
    parser.add_argument('--strategy', type=str, default="base",
                        help='strategy name.', choices=["base", "multihead", "transfer"])
    args = parser.parse_known_args()[0]
    # load dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    trainset, trainloader, valset, valloader, testset, testloader = datasets[args.dataset]()

    # load model
    backbone = resnet18(pretrained=True)
    classifier = torch.nn.Sequential(nn.Linear(1000, 512), nn.ReLU(),
                                     nn.Linear(512, trainset.num_sub_labels + trainset.num_coarse_labels))
    if args.strategy == "multihead":
        model = MultiHead(backbone=backbone, classifier=classifier)
        strategy = MultiHeadStrategy.MultiHeadStrategy(model=model, optimizer_name="Adam",
                                                       model_hparams={"num_classes": trainset.num_sub_labels + trainset.num_coarse_labels,
                                                                      "act_fn_name": "relu"},
                                                       optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
                                                       transfer=False)
    else:
        model = BaseModel(backbone=backbone, classifier=classifier)

        if args.strategy == "transfer":
            strategy = BaseStrategy.BaseStrategy(model=model, optimizer_name="Adam", model_hparams={"num_classes": trainset.num_sub_labels + trainset.num_coarse_labels,
                                                                                                    "act_fn_name": "relu"},
                                                 optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4}, transfer=True)
        elif args.strategy == "base":
            strategy = BaseStrategy.BaseStrategy(model=model, optimizer_name="Adam", model_hparams={"num_classes": trainset.num_sub_labels + trainset.num_coarse_labels,
                                                                                                    "act_fn_name": "relu"},
                                                 optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4}, transfer=False)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, strategy.name,args.dataset),  # Where to save models
                         gpus=1 if str(device) == "cuda:0" else 0,  # We run on a single GPU (if possible)
                         max_epochs=180,  # How many epochs to train for if no patience is set
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                    LearningRateMonitor("epoch"),
                                    TQDMProgressBar(1)]  # Log learning rate every epoch
                         )

    trainer.fit(strategy, train_dataloaders=trainloader, val_dataloaders=valloader)
    trainer.test(strategy, dataloaders=testloader)


if __name__ == '__main__':
    main()
