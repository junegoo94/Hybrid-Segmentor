import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights
import torch.optim as optim
import config
import torchvision.transforms as TF
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification \
    import BinaryJaccardIndex, BinaryRecall, BinaryAccuracy, \
        BinaryPrecision, BinaryF1Score, Dice

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(pl.LightningModule):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], learning_rate = config.LEARNING_RATE
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


        # loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Confusion matrix
        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()
        
        # Overlapped area metrics (Ignore Backgrounds)
        self.jaccard_ind = BinaryJaccardIndex()
        self.dice = Dice()

        # LR
        self.lr = learning_rate

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        y = torch.tensor(y, dtype=torch.int32)
        dice = self.dice(pred, y)

        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score, 
                      'train_precision': precision,  'train_recall': re, 'train_IOU': jaccard, 'train_dice': dice},
                      on_step=False, on_epoch=True, prog_bar=True)
        
        # if batch_idx % 100 == 0:
        #     x = x[:8]
        #     grid = torchvision.utils.make_grid(x.view(-1, 3, 256, 256))
        #     self.logger.experiment.add_image("crack_images", grid, self.global_step)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        y = torch.tensor(y, dtype=torch.int32)
        dice = self.dice(pred, y)

        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_f1_score': f1_score, 
                      'val_precision': precision,  'val_recall': re, 'val_IOU': jaccard, 'val_dice': dice},
                      on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        loss, pred, y = self._common_step(batch, batch_idx)
        loss, pred, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        y = torch.tensor(y, dtype=torch.int32)
        dice = self.dice(pred, y)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_f1_score': f1_score, 
                      'test_precision': precision,  'test_recall': re, 'test_IOU': jaccard, 'test_dice': dice},
                      on_step=False, on_epoch=True, prog_bar=False) 
        return loss
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(config.DEVICE)
        y = y.float().unsqueeze(1).to(config.DEVICE)
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        preds = torch.sigmoid(pred)
        preds = (preds > 0.5).float()
        return loss, preds, y
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(config.DEVICE)
        y = y.float().unsqueeze(1).to(config.DEVICE)
        pred = self.forward(x)
        preds = torch.sigmoid(pred)
        preds = (preds > 0.5).float()
        return preds
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()