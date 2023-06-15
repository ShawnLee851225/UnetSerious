from torchvision import transforms
from torch.nn.functional import softmax
import pandas as pd
import numpy as np
import torch
def show_predict_image(train_pred):
    pred_pic = softmax(train_pred[0],dim=1)
    pred_pic[pred_pic>=0.5]=1
    pred_pic[pred_pic<0.5]=0
    to_PIL = transforms.ToPILImage()
    pred_pil = to_PIL(pred_pic)
    pred_pil.show()
def count_confusion_matrix(train_pred,label) ->np:
    to_PIL = transforms.ToPILImage()
    metrics=np.zeros((2,2),dtype=np.int32)
    """
    TP=metrics[0][0]
    FP=metrics[0][1]
    FN=metrics[1][0]
    TN=metrics[1][1]
    """
    # train pred
    for k in range (len(train_pred)):
        pred_pic = softmax(train_pred[k],dim=1) # ->0~1
        pred_pic[pred_pic>=0.5]=1
        pred_pic[pred_pic<0.5]=0
 
        pred_pic = torch.as_tensor(pred_pic,dtype=torch.int32)

        label_pic = label[k] # 0~1
        label_pic[label_pic>=0.5]=1
        label_pic[label_pic<0.5]=0

        label_pic = torch.as_tensor(label_pic,dtype=torch.int32)
        for i in range(label_pic.size(1)):
            for j in range(label_pic.size(2)):
                lab = int(label_pic[0][i][j])
                pred =int(pred_pic[0][i][j])
                if( lab == 1 and pred == 1 ):
                    # TP
                    metrics[0][0] += 1
                elif( lab == 0 and pred == 0 ):
                    # TN
                    metrics[1][1] += 1
                elif( lab == 1 and pred == 0 ):
                    # FN
                    metrics[1][0] += 1
                elif( lab == 0 and pred == 1 ):
                    # FP
                    metrics[0][1] += 1
    return metrics


def count_IOU(metrics):
    """
    IOU = TP/TP+FP+FN
    """
    IOU = 0.0
    IOU =metrics[0][0] / ( metrics[0][0]+ metrics[0][1] + metrics[1][0] )
    return IOU
def count_PRF1(metrics):
    """
    Precision = TP/TP+FP
    Recall = TP/TP+FN
    F1-score = 2*Precision*Recall / (Precision + Recall)
    """
    Precision = metrics[0][0] / ( metrics[0][0] + metrics[0][1] )
    Recall = metrics[0][0] / ( metrics[0][0] + metrics[1][0] )
    F1_score = 2*Precision*Recall / (Precision + Recall)

    return Precision,Recall,F1_score

def list2excel(df, path,index) :
    df = pd.DataFrame(df).to_excel(path,index= index)