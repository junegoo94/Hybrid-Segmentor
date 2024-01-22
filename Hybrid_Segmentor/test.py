import torch
import os
from model import HybridSegmentor
from utils import save_predictions_as_imgs, eval_metrics, eval_ODS, eval_OIS
from dataloader import get_loaders
import config


def main():
    trian_loader, val_loader_loader, test_loader = get_loaders(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        config.TEST_IMG_DIR,
        config.TEST_MASK_DIR,
        1,
        config.NUM_WORKERS,
        config.PIN_MEMORY,
    )
    
    dataloaders = {'train': trian_loader, 'val': val_loader_loader, 'test': test_loader}
    model = HybridSegmentor().to(config.DEVICE)

    print('Loading Model')

    ck_file_path = r'/trinity/home/jmohgoo/data/junegoo/model/v7_resnet/checkpoints/v7_RECALL_final/best.ckpt'
    checkpoint = torch.load(ck_file_path)
    model.load_state_dict(checkpoint['state_dict'])
    mul_outputs = True
    mode = 'test'


    # print()
    # print('Computing Metrics')
    # eval_metrics(loader=dataloaders[mode], model=model, multiple_outputs=mul_outputs)
    # print('-----------------------------')

    print('Saving Images')
    file_name = 'RECALL_outputs'
    current_path = os.getcwd()
    if file_name not in os.listdir(os.path.join(current_path)):
        os.makedirs(file_name)
    save_predictions_as_imgs(dataloaders[mode], model, folder=file_name+"/", device=config.DEVICE, multiple_outputs=mul_outputs)
    print('Saved all images')
    
    
    
    # print('-----------------------------')
    # print('Computing ODS')
    # eval_ODS(loader=dataloaders[mode], model=model, multiple_outputs=mul_outputs)
    # print('-----------------------------')
    # print('Computing OIS')
    # eval_OIS(loader=dataloaders[mode], model=model, multiple_outputs=mul_outputs)





if __name__ == "__main__":
    main()