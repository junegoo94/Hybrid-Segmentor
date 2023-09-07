from dataloader import get_loaders
import pytorch_lightning as pl
import config
from model import HybridCrackNet
import torch
from callback import MyPrintingCallBack, EarlyStopping, checkpoint_callback, early_stopping
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="v7_BCEDICE_0_2_final")
    train_loader, val_loader, test_loader = get_loaders(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, config.VAL_IMG_DIR, config.VAL_MASK_DIR, config.TEST_IMG_DIR,
                                                        config.TEST_MASK_DIR, config.BATCH_SIZE, config.NUM_WORKERS, config.PIN_MEMORY,)   

    model = HybridCrackNet(learning_rate=config.LEARNING_RATE).to(config.DEVICE)
    trainer = pl.Trainer(logger=logger, accelerator="gpu", min_epochs=1, 
                         max_epochs=config.NUM_EPOCHS, precision='16-mixed',
                         callbacks=[checkpoint_callback, early_stopping])
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")