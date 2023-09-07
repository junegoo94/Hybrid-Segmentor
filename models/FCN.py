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
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification \
    import BinaryJaccardIndex, BinaryRecall, BinaryAccuracy, \
        BinaryPrecision, BinaryF1Score, Dice


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channel = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
vgg19 = models.vgg19(VGG19_Weights.IMAGENET1K_V1)
max_pool_to_change = [(4), (9), (18), (27)]
for i in max_pool_to_change:
    vgg19.features[i] = nn.AvgPool2d(2,2)
vgg19.classifier = Identity()
vgg19.avgpool = Identity()

class vgg_FCN(pl.LightningModule):
    def __init__(self, encoder=vgg19, *, in_channels=3, out_channels=1, learning_rate=config.LEARNING_RATE):
        super(vgg_FCN, self).__init__()

        self.encoder = encoder.features
        self.block1 = nn.Sequential(*self.encoder[:19]) # pool3
        self.block2 = nn.Sequential(*self.encoder[19:28]) # pool4
        self.block3 = nn.Sequential(*self.encoder[28:])

        self.conv6 = nn.Conv2d(512, 4096, 7, 1)
        self.dropout1 = nn.Dropout2d(p=0.4)
        self.conv7 = nn.Conv2d(4096, 4096, 1)
        self.dropout2 = nn.Dropout2d(p=0.4)
        self.conv8 = nn.Conv2d(4096, 2, 1)

        self.deconv1 = nn.ConvTranspose2d(2, 512, 14,  2)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 2,  2)
        self.deconv3 = nn.ConvTranspose2d(256, 1, 8,  8)


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
        b1 = self.block1(x)
        b2 = self.block2(b1)
        x = self.block3(b2)
        x = self.dropout1(self.conv6(x))
        x = self.dropout2(self.conv7(x))
        x = self.conv8(x)
        x = self.deconv1(x) + b2
        x = self.deconv2(x) + b1
        x = self.deconv3(x)

        return x
    
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



# def test():
#     x = torch.randn((1, 3, 256, 256))
#     model = vgg_FCN(vgg19)
#     preds = model(x)
    

# if __name__ == "__main__":
#     test()