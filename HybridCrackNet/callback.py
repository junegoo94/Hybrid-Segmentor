from pytorch_lightning.callbacks import EarlyStopping, Callback, ModelCheckpoint
import os

class MyPrintingCallBack(Callback):
    def __init__(self):
        super(MyPrintingCallBack, self).__init__()

    def on_train_start(self, trainer, pl_module):
        print("Start Training")

    def on_train_end(self, trainer, pl_module):
        print("Training is done")

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(os.getcwd(), 'checkpoints', 'v7_BCEDICE0_2_final'),
    filename='v7-epoch{epoch:02d}-val_loss{val_loss:.4f}',
    verbose=True,
    save_last=True,
    save_top_k=5,
    monitor='val_loss',
    mode='min'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=True,
    mode='min'
)
