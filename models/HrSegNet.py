import math

import paddle
# import paddle.nn as nn
from torch import nn
import torch
import torch.nn.functional as F

# from paddleseg.utils import utils
# from paddleseg.cvlibs import manager, param_init
# from paddleseg.models.layers.layer_libs import SyncBatchNorm
# import paddle.nn.functional as F

import pytorch_lightning as pl
import config

import torch.optim as optim
from torch.optim import lr_scheduler

from metric import DiceBCELoss, DiceLoss
import torchmetrics
from torchmetrics.classification \
    import BinaryJaccardIndex, BinaryRecall, BinaryAccuracy, \
        BinaryPrecision, BinaryF1Score, Dice
import numpy as np




# features
# 1. The size of the high-resolution path remains constant throughout the process
# 2. In order to reduce and flexibly control the computational cost, the channel of 
# the high-resolution path remains unchanged
# 3.  We use multiple segmentation heads, two of which are auxiliary segmentation
# heads used for auxiliary loss during training
# 4. The seg head is performed in two steps, instead of restoring to the 
# original resolution all at once

class SegBlock(nn.Module):
    def __init__(self, 
                 base=32,
                 stage_index=1):# stage_index=1,2,3. 
        super(SegBlock, self).__init__()

        #  Convolutional layer for high-resolution paths with constant spatial resolution and constant channel
        self.h_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=base, out_channels=base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU()
        )
        self.h_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=base, out_channels=base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU()
        )
        self.h_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=base, out_channels=base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU()
        )

        # sematic guidance path/low-resolution path
        if stage_index==1: #first stage, stride=2, spatial resolution/2, channel*2
            self.l_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=base, out_channels=base*int(math.pow(2, stage_index)), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base*int(math.pow(2, stage_index))),
                nn.ReLU()
            )
        elif stage_index==2: #second stage
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=base, out_channels=base*int(math.pow(2, stage_index)), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base*int(math.pow(2, stage_index))),
                nn.ReLU()
            )
        elif stage_index==3: 
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=base, out_channels=base*int(math.pow(2, stage_index)), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base*int(math.pow(2, stage_index))),
                nn.ReLU(),
                nn.Conv2d(in_channels=base*int(math.pow(2, stage_index)), out_channels=base*int(math.pow(2, stage_index)), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base*int(math.pow(2, stage_index))),
                nn.ReLU()
            )
        else:
            raise ValueError("stage_index must be 1, 2 or 3")
        self.l_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=base*int(math.pow(2, stage_index)), out_channels=base*int(math.pow(2, stage_index)), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base*int(math.pow(2, stage_index))),
            nn.ReLU()
        )
        self.l_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=base*int(math.pow(2, stage_index)), out_channels=base*int(math.pow(2, stage_index)), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base*int(math.pow(2, stage_index))),
            nn.ReLU()
        )

        self.l2h_conv1 = nn.Conv2d(in_channels=base*int(math.pow(2, stage_index)), out_channels=base, kernel_size=1, stride=1, padding=0)
        self.l2h_conv2 = nn.Conv2d(in_channels=base*int(math.pow(2, stage_index)), out_channels=base, kernel_size=1, stride=1, padding=0)
        self.l2h_conv3 = nn.Conv2d(in_channels=base*int(math.pow(2, stage_index)), out_channels=base, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        # out = []
        # out.append(self.h_conv3(self.h_conv2(self.h_conv1(x))))
        # out.append(self.l_conv3(self.l_conv2(self.l_conv1(x))))
        size = x.shape[2:]
        out_h1 = self.h_conv1(x) # high resolution path
        out_l1 = self.l_conv1(x) # low resolution path
        # print(out_l1.shape)
        out_l1_i = F.interpolate(out_l1, size=size, mode='bilinear', align_corners=True) # upsample
        out_hl1 = self.l2h_conv1(out_l1_i) + out_h1 # low to high

        out_h2 = self.h_conv2(out_hl1)
        out_l2 = self.l_conv2(out_l1)
        # print(out_l2.shape)
        out_l2_i = F.interpolate(out_l2, size=size, mode='bilinear', align_corners=True)
        out_hl2 = self.l2h_conv2(out_l2_i) + out_h2

        out_h3 = self.h_conv3(out_hl2)
        out_l3 = self.l_conv3(out_l2)
        # print(out_l3.shape)
        out_l3_i = F.interpolate(out_l3, size=size, mode='bilinear', align_corners=True)
        out_hl3 = self.l2h_conv3(out_l3_i) + out_h3
        return out_hl3

# seg head
class SegHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, aux_head=False):
        super(SegHead, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        if aux_head:
            self.con_bn_relu = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=interplanes, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(interplanes),
                nn.ReLU(),
            )
        else:
            self.con_bn_relu = nn.Sequential(
                nn.ConvTranspose2d(in_channels=inplanes, out_channels=interplanes, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(interplanes),
                nn.ReLU(),
            )
        self.conv = nn.Conv2d(in_channels=interplanes, out_channels=1, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.con_bn_relu(x)
        out = self.conv(x)
        return out


# If you need to use this model with paddleseg, you need to add it to the model library 
# using manager.MODELS.add_component()
# @manager.MODELS.add_component
class HrSegNetB64(pl.LightningModule):
    """
    The HrSegNet implementation based on PaddlePaddle.s

    Args:
        num_classes (int): The unique number of target classes.
        
        in_channels (int, optional): The channels of input image. Default: 3.

        base (int, optional): The base channel number of the model. Default: 48.
    """
    def __init__(self,
                 in_channels=3, # input channel
                 base=64, # base channel of the model, 
                 num_classes=2, # number of classes
                 pretrained=None,
                 learning_rate =  0.01):
        super(HrSegNetB64, self).__init__()
        self.base = base
        self.num_classed = num_classes
        self.pretrained = pretrained
        # Stage 1 and 2 constitute the stem of the model, which is mainly used to extract low-level features.
        # Meanwhile, stage1 and 2 reduce the input image to 1/2 and 1/4 of the original size respectively
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=base // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base // 2),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=base // 2, out_channels=base, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(),
        )

        self.seg1 = SegBlock(base=base, stage_index=1)
        self.seg2 = SegBlock(base=base, stage_index=2)
        self.seg3 = SegBlock(base=base, stage_index=3)

        self.aux_head1 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.aux_head2 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.head = SegHead(inplanes=base, interplanes=base, outplanes=num_classes)

        # self.init_weight()

        # loss function
        # self.loss_fn = DiceBCELoss()
        # self.loss_fn = Dice()
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
        logit_list = []
        h, w = x.shape[2:]
        # aux_head only used in training
        # if self.training:
        stem1_out = self.stage1(x)
        stem2_out = self.stage2(stem1_out)
        hrseg1_out = self.seg1(stem2_out)
        hrseg2_out = self.seg2(hrseg1_out)
        hrseg3_out = self.seg3(hrseg2_out)
        last_out = self.head(hrseg3_out)
        seghead1_out = self.aux_head1(hrseg1_out)
        seghead2_out = self.aux_head2(hrseg2_out)
        logit_list = [last_out, seghead1_out, seghead2_out]
        logit_list = [F.interpolate(logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logit_list]
        return  logit_list
        # else:
        #     stem1_out = self.stage1(x)
        #     stem2_out = self.stage2(stem1_out)
        #     hrseg1_out = self.seg1(stem2_out)
        #     hrseg2_out = self.seg2(hrseg1_out)
        #     hrseg3_out = self.seg3(hrseg2_out)
        #     last_out = self.head(hrseg3_out)
        #     logit_list = [last_out]
        #     logit_list = [F.interpolate(logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logit_list]
        #     return  logit_list
        
    # def init_weight(self):
    #     if self.pretrained is not None:
    #         utils.load_entire_model(self, self.pretrained)
    #     else:
    #         for m in self.sublayers():
    #                 if isinstance(m, nn.Conv2D):
    #                     param_init.kaiming_normal_init(m.weight)
    #                 elif isinstance(m, nn.BatchNorm2D):
    #                     param_init.constant_init(m.weight, value=1)
    #                     param_init.constant_init(m.bias, value=0)
        
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
        pred_lst = self.forward(x)

        pred_1 = pred_lst[0]
        pred_2 = pred_lst[1]
        pred_3 = pred_lst[2]
        loss_1 = self.loss_fn(pred_1, y)
        loss_2 = self.loss_fn(pred_2, y)
        loss_3 = self.loss_fn(pred_3, y)
        loss = loss_1 + 0.5*loss_2 + 0.5*loss_3
        # loss_recall = 1-self.recall(pred, y)
        # loss *= loss_recall
        pred_1 = torch.sigmoid(pred_1)
        pred_1 = (pred_1 > 0.5).float()
        return loss, pred_1, y
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(config.DEVICE)
        y = y.float().unsqueeze(1).to(config.DEVICE)
        pred = self.forward(x)
        preds = torch.sigmoid(pred)
        preds = (preds > 0.5).float()
        return preds
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        lr_schedule = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule,
                "monitor": "val_loss",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
