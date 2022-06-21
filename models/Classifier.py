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

def create_model(num_classes=10, pretrained=False):
    if (pretrained):
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = resnet18(pretrained=False, num_classes=num_classes)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class LitResnet(pl.LightningModule):
    def __init__(self, learning_rate=0.1,num_classes=10, pretrained=False, unc=False, mode='sup', batch_size=200):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model(num_classes, pretrained)
        self.activations = collections.defaultdict(list)  # a dictionary that keeps saving the activations as they come
        self.unc = unc
        self.n_aug = 5
        self.threshold = 80
        self.transforms = torch.nn.Sequential(
                                            T.RandomHorizontalFlip(p=0.8),
                                            T.RandomVerticalFlip(p=0.3),
        )
        self.unc_values = list()
        self.sup = list()
        self.mode = mode
        self.learning_rate=learning_rate
        self.batch_size= batch_size

    def save_activation(self, name, mod, inp, out):
        self.activations[name].append(out.cpu())

    def set_activations(self, save_activations=True):
        # a dictionary that keeps saving the activations as they come
        if save_activations:
            for name, m in self.model.named_modules():
                if name == 'avgpool':
                    # partial to assign the layer name to each hook
                    m.register_forward_hook(partial(self.save_activation, name))

    def unc_configuration(self, n_aug=5, threshold=80):
        self.n_aug = n_aug
        self.threshold = threshold

    def fine_tuning(self, dim_out=10, freeze=False):
        self.model.fc = nn.Linear(self.model.fc.in_features, dim_out)
        if (freeze):
            # Freeze all layers just the classifier
            for name, parameters in self.model.named_parameters():
                # We freeze all the layers but the normalization ones and the fully connected layer
                if ('bn' not in name) and ('fc' not in name):
                    parameters.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        if self.unc:
            return F.softmax(out, dim=1)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        if self.mode == 'sup':
            x, _, y = batch
        else:
            x, y, _ = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None, unc=None):
        if unc:
            x, _, y = batch                                          #si calcola l'incertezza per le superclassi
            aug = x.repeat(self.n_aug, 1, 1, 1)
            for i in range(self.n_aug):
                aug[i] = self.transforms(x)
            logits = self(aug)
            logits = torch.sum(logits, dim=0) / self.n_aug            #sono già probabilità
            pred, ind = torch.max(logits, dim=0)
            if ind == y:
                unc = 1 - pred
            else:
                unc = 1 + pred
            self.unc_values.append(unc.cpu())
            self.sup.append(y.cpu())
        else:
            if self.mode == 'sup':
                x, _, y = batch
            else:
                x, y, _ = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, y)
            if stage:
                self.log(f"{stage}_loss", loss, prog_bar=True)
                self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        if(self.unc):
            self.evaluate(batch, "test_unc", self.unc)
        else:
            self.evaluate(batch, "test")

#    def configure_optimizers(self):
 #       optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True)
  #      scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0.00001, last_epoch=-1, verbose=True)
   #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.000001,
                                                                         last_epoch=-1)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,8,16,32], gamma=0.2)
        return [optimizer], [scheduler]