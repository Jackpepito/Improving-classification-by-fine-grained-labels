# torch
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
import collections
from functools import partial
import torchvision.transforms as T
from torchmetrics.functional import accuracy

# pytorch lighting
import pytorch_lightning as pl

def create_model(num_classes=10, num_superclasses=2, pretrained=False):
    if (pretrained):
        model = resnet18(pretrained=True)
    else:
        model = resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    clsub = nn.Linear(model.fc.in_features, num_classes)
    clsup = nn.Linear(model.fc.in_features, num_superclasses)
    model.fc = nn.Identity()
    print(model)
    print(clsup)
    print(clsub)
    return model, clsub, clsup

class MultiHead(pl.LightningModule):

    def __init__(self, learning_rate=0.1,num_classes=10, num_superclasses=2 ,pretrained=False, alpha=0.5):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.num_classes=num_classes
        self.num_superclasses=num_superclasses
        self.pretrained=pretrained
        # Create model
        self.model, self.clsub, self.clsup = create_model(self.num_classes,self.num_superclasses, self.pretrained)
        self.alpha = alpha
        self.learning_rate = learning_rate

    def forward(self, x):
        z=self.model(x)
        logits_sub = self.clsub(z)
        logits_sup = self.clsup(z)
        return F.log_softmax(logits_sub, dim=1), F.log_softmax(logits_sup, dim=1)

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, suby, supy = batch
        logits_sub, logits_sup = self.forward(imgs)
        loss_sup = F.nll_loss(logits_sup, supy.to('cuda'))
        loss_sub = F.nll_loss(logits_sub, suby.to('cuda'))
        loss = self.alpha*loss_sup + (1 - self.alpha)*loss_sub
        correct_sup = (logits_sup.argmax(1) == supy).float().mean()
        correct_sub = (logits_sub.argmax(1) == suby).float().mean()
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc_sup', correct_sup, on_step=False, on_epoch=True)
        self.log('train_acc_sub', correct_sub, on_step=False, on_epoch=True)
        self.log('train_loss_sup', loss_sup)
        self.log('train_loss_sub', loss_sub)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None, unc=None):
        imgs, _, supy = batch
        _, ypredsup = self.forward(imgs)
        loss_sup = F.nll_loss(ypredsup, supy.to('cuda'))
        correct_sup = (ypredsup.argmax(1) == supy).float().mean()
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log(f"{stage}_loss sup", loss_sup, prog_bar=True)
        self.log(f"{stage}_acc sup", correct_sup, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.000001,
                                                                         last_epoch=-1, verbose=True)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8, 16, 32], gamma=0.2)
        return [optimizer], [scheduler]