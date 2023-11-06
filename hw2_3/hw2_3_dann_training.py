'''
參考：https://github.com/NaJaeMin92/pytorch_DANN
且直接下面連結的model架構(層數、filter數目等都一樣)：
https://github.com/pha123661/NTU-2022Fall-DLCV/blob/master/HW2/P3_USPS_model.py 
'''
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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


class dg_dataset(Dataset):
    def __init__(self, img_path, tfm, label_csv_path) -> None:
        super(dg_dataset).__init__()
        self.img_path = img_path
        self.filenames = list()
        self.labels = list()
        self.transform = tfm

        with open(label_csv_path, mode='r') as file:
            reader = csv.reader(file)
            next(iter(reader))
            for row in reader:
                self.filenames.append(row[0])
                self.labels.append(int(row[1]))
        
        self.filenames = [os.path.join(self.img_path, x) for x in self.filenames]
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        im = Image.open(fname).convert('RGB')
        im = self.transform(im)
        label = self.labels[idx]

        return im, label
    
tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        ]) 



digit_dataset_train_source = dg_dataset(img_path= '../hw2_data/digits/mnistm/data', tfm= tf, label_csv_path = '../hw2_data/digits/mnistm/train.csv')
digit_dataset_val_source = dg_dataset(img_path= '../hw2_data/digits/mnistm/data', tfm= tf, label_csv_path = '../hw2_data/digits/mnistm/val.csv')

svhn_train_target = dg_dataset(img_path= '../hw2_data/digits/svhn/data', tfm= tf, label_csv_path = '../hw2_data/digits/svhn/train.csv')
usps_train_target = dg_dataset(img_path= '../hw2_data/digits/usps/data', tfm= tf, label_csv_path = '../hw2_data/digits/usps/train.csv')

svhn_val_target = dg_dataset(img_path= '../hw2_data/digits/svhn/data', tfm= tf, label_csv_path = '../hw2_data/digits/svhn/val.csv')
usps_val_target = dg_dataset(img_path= '../hw2_data/digits/usps/data', tfm= tf, label_csv_path = '../hw2_data/digits/usps/val.csv')

train_target = [svhn_train_target, usps_train_target]
val_target = [svhn_val_target, usps_val_target]
F = FeatureExtractor()
L = LabelPredictor()
D = DomainClassifier()
print(F)
print(L)
print(D)
for index in range(2):
    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a model, and put it on the device specified.
    F = FeatureExtractor().to(device)
    L = LabelPredictor().to(device)
    D = DomainClassifier().to(device)

    # The number of batch size.
    batch_size = 1024

    # The number of training epochs.
    n_epochs = 1

    # weight decay
    wd_num = 0.00001

    # MixUp alpha(α \alphaα在0.2 ~ 2之間效果都差不多，表示mixup對α \alphaα參數並不是很敏感。但如果α \alphaα過小，等於沒有進行mixup的原始數據，如果α \alphaα過大，等於所有輸入都是各取一半混合)
    alpha = 0

    # If no improvement in 'patience' epochs, early stop.
    patience = 200

    # For the classification task, we use cross-entropy as the measurement of performance.
    label_loss_criterion = nn.CrossEntropyLoss()
    domain_loss_criterion = nn.BCEWithLogitsLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(chain(F.parameters(), L.parameters(), D.parameters()), lr=0.0003, weight_decay=wd_num)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10)

    import os
    ckpt_dir = f"./dann_{index}"
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir) #創建空的目錄

    source_train_loader = DataLoader(
        digit_dataset_train_source, batch_size, shuffle=True, num_workers=0)
    target_train_loader = DataLoader(
        train_target[index], batch_size, shuffle=True, num_workers=0)
    target_val_loader = DataLoader(
        val_target[index], 2 * batch_size, shuffle=False, num_workers=0)
    
    target_train_loader = iter(cycle(target_train_loader))

    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_acc = 0
    step_num = 0
    total_num = n_epochs*len(source_train_loader)
    for epoch in range(n_epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        F.train()
        L.train()
        D.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for (src_imgs, src_labels),(tgt_imgs, _) in tqdm(zip(source_train_loader, target_train_loader)):

            # A batch consists of image data and corresponding labels.
            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)
            
            lambda_ = 2.0 / (1.0 + math.exp(-10 * (step_num/total_num))) - 1

            # 計算在source domain的分類損失
            # Forward the data. (Make sure data and model are on the same device.)
            # 提取特徵
            src_feature = F(src_imgs)
            # label classification loss
            src_logits = L(src_feature)
            label_loss = label_loss_criterion(src_logits, src_labels)

            # 計算在target domain的二元分類損失
            # Forward the data. (Make sure data and model are on the same device.)
            # 提取特徵
            all_img = torch.cat([src_imgs, tgt_imgs], dim=0)
            all_feature = F(all_img)
            # label classification loss
            target_logits = D(all_feature, lambda_)
            target_logits = target_logits.squeeze()
            target_labels = torch.cat([
            # source=0
            torch.zeros(src_imgs.shape[0], dtype=torch.float),
            # target=1
            torch.ones(tgt_imgs.shape[0], dtype=torch.float)
            ], dim=0).to(device)

            target_loss = domain_loss_criterion(target_logits, target_labels)

            loss = label_loss + target_loss
            
            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Update the parameters with computed gradients.
            optimizer.step()
            step_num += 1

            # Compute the accuracy for current batch.
            acc = (src_logits.argmax(dim=-1) == src_labels).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        F.eval()
        L.eval()
        D.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

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
                tgt_logits = L(tgt_feature)

            # We can still compute the loss (but not the gradient).
            loss = label_loss_criterion(tgt_logits, tgt_labels)

            # Compute the accuracy for current batch.
            acc = (tgt_logits.argmax(dim=-1) == tgt_labels).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            #break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        scheduler.step(valid_acc)
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # update logs
        if valid_acc > best_acc:
            with open("./sample_best_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open("./sample_best_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save models
        if valid_acc > best_acc:
            best_epoch = epoch+1
            print(f"Best model found at epoch {best_epoch}, saving model")
            torch.save(F.state_dict(), os.path.join(ckpt_dir,'feature_extractor')) # only save best to prevent output memory exceed error
            torch.save(L.state_dict(), os.path.join(ckpt_dir,'classifier'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir,'DomainClassifier'))
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping at no.{epoch+1} epoch, best epoch at {best_epoch}.")
                break