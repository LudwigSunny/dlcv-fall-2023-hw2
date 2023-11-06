'''
參考：https://github.com/NaJaeMin92/pytorch_DANN
且直接下面連結的model架構(層數、filter數目等都一樣)：
https://github.com/pha123661/NTU-2022Fall-DLCV/blob/master/HW2/P3_USPS_model.py 
'''
from itertools import chain
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import csv
import os
from PIL import Image
import math
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='Description of my script.')

parser.add_argument('--testing_img_dir', type=str, help='path to the directory of predefined noises')
parser.add_argument('--output_file_path', type=str, help='path to the directory for your 10 generated images ')

# 解析命令行参数
args = parser.parse_args()

# storing the arguments
testing_img_dir = args.testing_img_dir
output_file_path = args.output_file_path

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class GRF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.view_as(x)  # NECESSARY! autograd checks if tensor is modified

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.saved_variables[0]
        return grad_output.neg() * lambda_, None


class FeatureExtractor(nn.Module):
    def __init__(self, in_chans=3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=3),
        )

    def forward(self, x):
        feature = self.conv(x)
        feature = feature.reshape(-1, 128)
        return feature


class LabelPredictor(nn.Module):
    def __init__(self, n_classes=10) -> None:
        super().__init__()
        self.l_clf = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.l_clf(x)
        return x


class DomainClassifier(nn.Module):
    '''
    A Binary classifier
    '''

    def __init__(self) -> None:
        super().__init__()
        self.d_clf = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1),
        )

    def forward(self, x, lambda_):
        x = GRF.apply(x, lambda_)
        x = self.d_clf(x)
        return x


class test_dataset(Dataset):
    def __init__(self, img_path, tfm) -> None:
        super(test_dataset).__init__()
        self.img_path = img_path
        self.filenames = sorted([x for x in os.listdir(self.img_path) if x.endswith(".png")])
        self.transform = tfm
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = os.path.join(self.img_path, self.filenames[idx])
        im = Image.open(fname).convert('RGB')
        im = self.transform(im)

        return im, -1
    
    def get_file_names(self):
        return self.filenames
    
tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        ]) 


#testing_img_dir = '../hw2_data/digits/svhn/data'
#output_file_path = './svhn_pred.csv'
if 'svhn'.casefold() in testing_img_dir:
    F_ckpt = './hw2_3/ckpt/svhn_feature_extractor'
    L_ckpt = './hw2_3/ckpt/svhn_classifier'
if 'usps'.casefold() in testing_img_dir:
    F_ckpt = './hw2_3/ckpt/usps_feature_extractor'
    L_ckpt = './hw2_3/ckpt/usps_classifier'

val_target = test_dataset(img_path= testing_img_dir, tfm= tf)


# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
F = FeatureExtractor()
L = LabelPredictor()
F.load_state_dict(torch.load(F_ckpt))
L.load_state_dict(torch.load(L_ckpt))

F.to(device)
L.to(device)

# The number of batch size.
batch_size = 1024
target_val_loader = DataLoader(val_target, batch_size, shuffle=False, num_workers=0)


# ---------- Validation ----------
# Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
F.eval()
L.eval()
prediction = []
# Iterate the validation set by batches.
for batch in tqdm(target_val_loader):

    # A batch consists of image data and corresponding labels.
    tgt_imgs, tgt_labels = batch
    tgt_imgs, tgt_labels = tgt_imgs.to(device), tgt_labels.to(device)

    # We don't need gradient in validation.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        # 提取特徵
        tgt_feature = F(tgt_imgs)
        # label classification loss
        tgt_logits = L(tgt_feature).cpu().data.numpy()

        test_label = np.argmax(tgt_logits, axis=1)
        prediction += test_label.squeeze().tolist()

#create test csv
filename = val_target.get_file_names()
df = pd.DataFrame()
df["image_name"] = filename
df["label"] = prediction
df.to_csv(output_file_path, index = False)