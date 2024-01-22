import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

result_save_ind = 0
threshold = 0.5

def eval_metrics(loader, model, device="cuda", multiple_outputs=False):
    model.eval()
    eps = 1e-7

    TP_total = 0
    FP_total = 0
    TN_total = 0
    FN_total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            if multiple_outputs == True:
                final_output = model(x)[result_save_ind]
                preds_probability = torch.sigmoid(final_output)
                preds = (preds_probability > threshold).float()
            else:
                preds_probability = torch.sigmoid(model(x))
                preds = (preds_probability > threshold).float()

            confusion_matirx = preds / y

            TP =  torch.sum(confusion_matirx == 1).item()
            FP = torch.sum(confusion_matirx == float('inf')).item()
            TN = torch.sum(torch.isnan(confusion_matirx)).item()
            FN = torch.sum(confusion_matirx == 0).item()


            TP_total += TP
            FP_total += FP
            TN_total += TN
            FN_total += FN


    accuracy = (TP_total + TN_total) / (TP_total + FP_total + TN_total + FN_total + eps)
    precision = (TP_total) / (TP_total+FP_total + eps)
    recall = (TP_total) / (TP_total+FN_total + eps) # TP rate
    FP_rate = FP_total / (FP_total+TN_total + eps)
    f1_score = 2* (precision*recall)/(precision+recall+eps)
    dice_score = 2*TP_total / (2*TP_total+FP_total+FN_total + eps) # will be the same as f1 score
    IOU_score = TP_total / (TP_total + FP_total + FN_total + eps)

    print(f'Global Accuracy : {accuracy} / Precision : {precision} / Recall : {recall} / FPR : {FP_rate} / F1 score : {f1_score}')
    print(f'Dice Score {dice_score} / IOU score {IOU_score}')

def eval_OIS(loader, model, device="cuda", multiple_outputs=False):
    best_OIS_lst = []
    best_thres_lst = []
    thres_list = [i for i in np.arange(0, 1, step=0.01)]
    eps = 1e-7

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            best_thres = 0
            best_OIS = 0
            for thres in thres_list:
                if multiple_outputs == True:
                    final_output = model(x)[result_save_ind]
                    preds_probability = torch.sigmoid(final_output)
                    preds = (preds_probability > threshold).float()
                else:
                    preds_probability = torch.sigmoid(model(x))
                    preds = (preds_probability > threshold).float()


                confusion_matirx = preds / y

                TP = torch.sum(confusion_matirx == 1).item()
                FP = torch.sum(confusion_matirx == float('inf')).item()
                TN = torch.sum(torch.isnan(confusion_matirx)).item()
                FN = torch.sum(confusion_matirx == 0).item()

                precision = (TP) / (TP+FP+eps)
                recall = (TP) / (TP+FN+eps) # TP rate
                f1_score = 2* (precision*recall)/(precision+recall+eps)

                if f1_score > best_OIS:
                    best_OIS = f1_score
                    best_thres = thres

            best_thres_lst.append(best_thres)
            best_OIS_lst.append(best_OIS)

    mean_OIS = np.mean(best_OIS_lst)
    mean_thres = np.mean(best_thres_lst)

    print(f'OIS F1 Score : {mean_OIS} / with the mean threshod : {mean_thres}')

    return mean_OIS, mean_thres

def eval_ODS(loader, model, device="cuda", multiple_outputs=False):
    model.eval()

    best_ODS = 0
    best_thres = 0
    thres_list = [i for i in np.arange(0, 1, step=0.01)]
    eps = 1e-7


    with torch.no_grad():
        for thres in thres_list:
            TP_total = 0
            FP_total = 0
            TN_total = 0
            FN_total = 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)

                if multiple_outputs == True:
                    final_output = model(x)[result_save_ind]
                    preds_probability = torch.sigmoid(final_output)
                    preds = (preds_probability > threshold).float()
                else:
                    preds_probability = torch.sigmoid(model(x))
                    preds = (preds_probability > threshold).float()

                confusion_matirx = preds / y

                TP =  torch.sum(confusion_matirx == 1).item()
                FP = torch.sum(confusion_matirx == float('inf')).item()
                TN = torch.sum(torch.isnan(confusion_matirx)).item()
                FN = torch.sum(confusion_matirx == 0).item()

                TP_total += TP
                FP_total += FP
                TN_total += TN
                FN_total += FN

            precision = (TP_total) / (TP_total+FP_total+eps)
            recall = (TP_total) / (TP_total+FN_total+eps) # TP rate
            f1_score = 2* (precision*recall)/(precision+recall+eps)
            if f1_score > best_ODS:
                best_ODS = f1_score
                best_thres = thres

    print(f'ODS F1 Score : {best_ODS} / with the threshod : {best_thres}')

    return best_ODS, best_thres

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", multiple_outputs=False):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            if multiple_outputs == True:
                final_output = model(x)[result_save_ind]
                preds_probability = torch.sigmoid(final_output)
                preds = (preds_probability > threshold).float()
                # preds = preds_probability
            else:
                preds_probability = torch.sigmoid(model(x))
                preds = (preds_probability > threshold).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()



def loss_plot(train_loss, val_loss):
    if len(train_loss) != len(val_loss):
        print('The number of losses are different')
    else:
        labels = [i for i in range(1, len(train_loss)+1)]
        plt.plot(train_loss)
        plt.plot(val_loss)
        # plt.xticks(range(0, len(train_loss), 10), labels[::9])
        ticks = [i for i in range(0, len(train_loss), 10)]  # Ticks at every 10th index
        tick_labels = [labels[i-1] for i in ticks]  # Corresponding labels for the ticks
        plt.xticks(ticks, tick_labels)
        plt.gca().get_xticklabels()[0].set_visible(False)
        plt.xlabel('Epoch', fontsize=17)
        plt.ylabel('Loss', fontsize=17)
        plt.show()
        plt.savefig('loss_output.png')