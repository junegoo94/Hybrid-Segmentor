from torch import nn
import torch
import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange
from math import sqrt
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
import torchmetrics
import torchmetrics as Metric
import torch.optim as optim
from torch.optim import lr_scheduler
import config
from dataloader import get_loaders
import torchvision
import os
from metric import FocalLoss, DiceLoss, IoULoss
from torchmetrics.classification \
    import BinaryJaccardIndex, BinaryRecall, BinaryAccuracy, \
        BinaryPrecision, BinaryF1Score, Dice


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Down(nn.Module):

    def __init__(self, nn):
        super(Down,self).__init__()
        self.nn = nn
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        down = self.nn(inputs)
        unpooled_shape = down.size()
        outputs, indices = self.maxpool_with_argmax(down)
        return outputs, down, indices, unpooled_shape

class Up(nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        outputs = self.nn(outputs)
        return outputs

class Fuse(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(64,1)

    def forward(self,down_inp,up_inp):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode='bilinear')
        outputs = self.nn(outputs)

        return self.conv(outputs)



class DeepCrack(pl.LightningModule):

    def __init__(self, learning_rate = config.LEARNING_RATE):
        super(DeepCrack, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            ConvRelu(3,64),
            ConvRelu(64,64),
        ))

        self.down2 = Down(torch.nn.Sequential(
            ConvRelu(64,128),
            ConvRelu(128,128),
        ))

        self.down3 = Down(torch.nn.Sequential(
            ConvRelu(128,256),
            ConvRelu(256,256),
            ConvRelu(256,256),
        ))

        self.down4 = Down(torch.nn.Sequential(
            ConvRelu(256, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.down5 = Down(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.up1 = Up(torch.nn.Sequential(
            ConvRelu(64, 64),
            ConvRelu(64, 64),
        ))

        self.up2 = Up(torch.nn.Sequential(
            ConvRelu(128, 128),
            ConvRelu(128, 64),
        ))

        self.up3 = Up(torch.nn.Sequential(
            ConvRelu(256, 256),
            ConvRelu(256, 256),
            ConvRelu(256, 128),
        ))

        self.up4 = Up(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 256),
        ))

        self.up5 = Up(torch.nn.Sequential(
            ConvRelu(512, 512),
            ConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.fuse5 = Fuse(ConvRelu(512 + 512, 64), scale=16)
        self.fuse4 = Fuse(ConvRelu(512 + 256, 64), scale=8)
        self.fuse3 = Fuse(ConvRelu(256 + 128, 64), scale=4)
        self.fuse2 = Fuse(ConvRelu(128 + 64, 64), scale=2)
        self.fuse1 = Fuse(ConvRelu(64 + 64, 64), scale=1)

        self.final = Conv3X3(5,1)

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

    

    def forward(self,inputs):

        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)

        # decoder part
        up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)

        fuse5 = self.fuse5(down_inp=down5,up_inp=up5)
        fuse4 = self.fuse4(down_inp=down4, up_inp=up4)
        fuse3 = self.fuse3(down_inp=down3, up_inp=up3)
        fuse2 = self.fuse2(down_inp=down2, up_inp=up2)
        fuse1 = self.fuse1(down_inp=down1, up_inp=up1)

        output = self.final(torch.cat([fuse5,fuse4,fuse3,fuse2,fuse1],1))

        return output, fuse5, fuse4, fuse3, fuse2, fuse1
    



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
        loss_output = self.loss_fn(pred[0], y)
        loss_5 = self.loss_fn(pred[1], y)
        loss_4 = self.loss_fn(pred[2], y)
        loss_3 = self.loss_fn(pred[3], y)
        loss_2 = self.loss_fn(pred[4], y)
        loss_1 = self.loss_fn(pred[5], y)
        preds = torch.sigmoid(pred[0])
        preds = (preds > 0.5).float()
        loss = loss_output + loss_1 + loss_2 + loss_3 + loss_4 + loss_5
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



# if __name__ == '__main__':
#     inp = torch.randn((1,3,256,256))

#     model = DeepCrack()

#     out = model(inp)
#     print(out[0].shape)

