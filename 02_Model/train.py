import os
import tqdm
import cv2
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models import CNNModel, VITModel
from dataset import make_loader
from configures import CFG


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_fn(data_loader, model, criterion, epoch_loss = 0.0):
    model.train()

    preds = []
    actuals = []

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])

    for i_batch, item in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        images = item['image'].to(CFG['DEVICE'])
        labels = item['label'].to(CFG['DEVICE'])

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds.extend(predicted.tolist())  # 예측값을 리스트에 추가
        actuals.extend(labels.tolist())  # 실제값을 리스트에 추가

        epoch_loss += loss.item()

    return epoch_loss, preds, actuals

def eval_fn(data_loader, model, criterion, epoch_loss = 0.0):
    model.eval()
    criterion.eval()

    preds = []
    actuals = []

    with torch.no_grad():
        for i_batch, item in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            images = item['image'].to(CFG['DEVICE'])
            labels = item['label'].to(CFG['DEVICE'])

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            val_loss = criterion(outputs, labels)

            epoch_loss += val_loss.item()

            preds.extend(predicted.tolist())  # Add predicted values to the list
            actuals.extend(labels.tolist())  # Add actual values to the list

    return epoch_loss, preds, actuals


def run(model, train_loader, val_loader, model_name):
    wandb.init(project='OpenCrop-Pests-Classification', entity='jyihaan4', name=f'pests-classification tomato {model_name}')

    wandb.config = {
        "learning_rate": CFG['LEARNING_RATE'],
        "epochs": CFG['EPOCHS'],
        "batch_size": CFG['BATCH_SIZE']
    }


    model_dir = f"../Output/{model_name}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)


    model = model.to(CFG['DEVICE'])

    criterion = nn.CrossEntropyLoss()

    for epoch in range(CFG['EPOCHS']):
        # train
        train_epoch_loss, train_preds, train_actuals = train_fn(train_loader, model, criterion)

        # valid
        val_epoch_loss, val_preds, val_actuals = eval_fn(val_loader, model, criterion)

        # acc, loss
        train_acc = accuracy_score(train_actuals, train_preds)
        train_loss = train_epoch_loss / len(train_loader)

        val_acc = accuracy_score(val_actuals, val_preds)
        val_loss = val_epoch_loss / len(val_loader)

        print(f'Epoch [{epoch + 1}/{CFG["EPOCHS"]}], '
              f'Train Loss: {train_loss}, '
              f'Train Accuracy: {train_acc}, '
              f'Val Loss: {val_loss}, '
              f'Val Accuracy: {val_acc}')

        wandb.log({"Train Accuracy": train_acc,
                   "Train Loss": train_loss,
                   "Val Accuracy": val_acc,
                   "Val Loss": val_loss})


        torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}_{epoch + 1}.pth'))

def main():
    print(CFG['DEVICE'])

    train_data = pd.read_csv("../Output/train_data.csv")

    train, val = train_test_split(train_data, test_size=0.2, random_state=CFG['SEED'])

    train_loader = make_loader(train, batch_size=CFG['BATCH_SIZE'],shuffle=True)
    val_loader = make_loader(val, batch_size=CFG['BATCH_SIZE'], shuffle=False)

    num_classes = train_data['label'].unique().max() + 1
    print('num_classes: ', num_classes)

    cnn_model = CNNModel(num_classes)
    vit_model = VITModel(num_classes)

    # train
    # cnn
    # run(cnn_model, train_loader, val_loader, model_name='cnn_aug')
    # vit
    run(vit_model, train_loader, val_loader, model_name='vit_aug')


if __name__ == '__main__':
    main()