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
import torchvision
from metric import DiceBCELoss
import torchmetrics
from torchmetrics.classification \
    import BinaryJaccardIndex, BinaryRecall, BinaryAccuracy, \
        BinaryPrecision, BinaryF1Score, Dice



DEVICE = config.DEVICE

# Layer Normalisation
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    
# Depth-wise CNN
class DepthWiseConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, padding, stride=1, bias=True):
        super(DepthWiseConv, self).__init__()
        # Depthwise Convolution
        self.DW_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
                                 kernel_size=kernel, stride=stride, 
                                 padding=padding, groups=in_dim, bias=bias)
        # Pointwise Convolution
        self.PW_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                 kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.DW_conv(x)
        x = self.PW_conv(x)

        return x
        
class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DoubleConv, self).__init__()

        hidden_dim = int((in_dim + out_dim)/2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        output = self.conv_block(x)

        return output

class OverlapPatchEmbedding(nn.Module):
    def __init__(self, kernel, stride, padding, in_dim, out_dim):
        super(OverlapPatchEmbedding, self).__init__()
        self.overlap_patches = nn.Unfold(kernel_size=kernel, stride=stride, padding=padding)
        self.embedding = nn.Conv2d(in_dim*kernel**2, out_dim, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.overlap_patches(x)
        n_patches = x.shape[-1]
        divider = int(sqrt(h*w / n_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h = h//divider)
        x = self.embedding(x)

        return x

class EfficientMSA(nn.Module):
    # same size of input and output
    def __init__(self, dim, n_heads, reduction_ratio):
        super(EfficientMSA, self).__init__()
        self.reshaping_k = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.reshaping_v = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        n, c, h, w = x.shape
        LN = LayerNorm2d(c).to(device=DEVICE)
        x = LN(x)
        reshaped_k = self.reshaping_k(x)
        reshaped_v = self.reshaping_v(x)
        reshaped_k = rearrange(reshaped_k, "b c h w -> b (h w) c") # reshape (batch, sequence_length, channels) for attention
        reshaped_v = rearrange(reshaped_v, "b c h w -> b (h w) c") # reshape (batch, sequence_length, channels) for attention
        q = rearrange(x, "b c h w -> b (h w) c")
        output, output_weights = self.attention(q, reshaped_k, reshaped_v)
        output = rearrange(output, "b (h w) c -> b c h w", h=h, w=w)

        return output


class MixFFN(nn.Module):
    # same size of inputs and outputs
    def __init__(self, dim, expansion_factor):
        super(MixFFN, self).__init__()
        latent_dim = dim*expansion_factor
        self.mixffn = nn.Sequential(
            nn.Conv2d(dim, latent_dim, 1),
            DepthWiseConv(latent_dim, latent_dim, kernel=3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_dim, dim, 1)
        )
    def forward(self, x):
        n, c, h, w = x.shape
        LN = LayerNorm2d(c).to(device=DEVICE)
        x = LN(x)
        x = self.mixffn(x)
        return x
    
class MiT(nn.Module):
    def __init__(self, channels, dims, n_heads, expansion, reduction_ratio, n_layers):
        super(MiT, self).__init__()
        kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (in_dim, out_dim), (kernel, stride, padding), n_layers, expansion, n_heads, reduction_ratio in zip(dim_pairs, kernel_stride_pad, n_layers, expansion, n_heads, reduction_ratio):
            overlapping = OverlapPatchEmbedding(kernel, stride, padding, in_dim, out_dim)
            layers = nn.ModuleList([])
            
            for _ in range(n_layers):
                layers.append(nn.ModuleList([EfficientMSA(dim=out_dim, n_heads=n_heads, reduction_ratio=reduction_ratio),
                              MixFFN(dim=out_dim, expansion_factor=expansion)]))
            self.stages.append(nn.ModuleList([overlapping, layers]))

    def forward(self, x):
        # h, w = x.shape[-2:]
        layer_outputs = []
        for overlapping, layers in self.stages:
            x = overlapping(x)  # (b, c x kernel x kernel, num_patches)
            for (attension, ffn) in layers:  # attention, feed forward
                x = attension(x) + x  # skip connection
                x = ffn(x) + x

            layer_outputs.append(x)  # multi scale features

        return layer_outputs
    
class conv_upsample(nn.Module):
    def __init__(self, scale, in_dim, out_dim=32):
        super(conv_upsample, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 1)
        self.upscale = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)

    def forward(self, x):
        output = self.upscale(self.conv(x))
        return output
        

    
class SegFormer(pl.LightningModule):
    def __init__(self, channels=3, dims=(64, 128, 320, 512), n_heads=(1, 2, 5, 8), 
                 expansion=(8, 8, 4, 4), reduction_ratio=(8, 4, 2, 1), 
                 n_layers=(3, 3, 18, 3), decoder_dim=256, learning_rate = 0.00006):
        
        super(SegFormer, self).__init__()
        self.mix_transformer = MiT(channels, dims, n_heads, expansion, reduction_ratio, n_layers)

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),  # First, multi-level features Fi from the MiT encoder go through an MLP layer to unify the channel dimension.
            nn.Upsample(scale_factor=2**(i+2))  # second step, features are up-sampled to 1/4th and concatenated together.
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),  # Third, a MLP layer is adopted to fuse the concatenated features
            nn.Conv2d(decoder_dim, 1, 1),  # Finally, another MLP layer takes the fused feature to predict the segmentation mask M with a H4 × W4 × Ncls resolution
        )

        # loss function
        # self.loss_fn = DiceBCELoss()
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
        mit_results = self.mix_transformer(x)

        
        fused = [to_fused(output) for output, to_fused in zip(mit_results, self.to_fused)]
        fused = torch.cat(fused, dim=1)
        return self.to_segmentation(fused)

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
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        lr_schedule = lr_scheduler.PolynomialLR(optimizer, power=1.0)
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

