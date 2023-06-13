from torchvision import transforms
from torch.nn.functional import softmax

def show_predict_image(train_pred):
    pred_pic = softmax(train_pred[0],dim=1)
    pred_pic *= 255
    to_PIL = transforms.ToPILImage()
    pred_pil = to_PIL(pred_pic)
    pred_pil.show()