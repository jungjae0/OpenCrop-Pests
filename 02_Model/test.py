import os
import tqdm
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models import CNNModel, VITModel
from dataset import make_loader
from configures import CFG


def test_fn(model, test_loader):
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0

        preds = []
        actuals = []
        paths = []

        for item in tqdm.tqdm(test_loader):
            images = item['image']
            labels = item['label']
            path = item['path']

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += len(labels)
            correct += (predicted == labels).sum().item()

            preds.extend(predicted.tolist())  # 예측값을 리스트에 추가
            actuals.extend(labels.tolist())  # 실제값을 리스트에 추가
            paths.extend(path)

        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))

    return preds, actuals, paths


def result_df(preds, actuals, paths):

    # label_mapping = {
    #     0: "정상",
    #     1: "검거세미밤나방",
    #     2: "꽃노랑총채벌레",
    #     3: "담배가루이",
    #     4: "담배거세미나방",
    #     5: "담배나방",
    #     6: "도둑나방",
    #     7: "먹노린재",
    #     8: "목화바둑명나방",
    #     9: "무잎벌",
    #     10: "배추좀나방",
    #     11: "배추흰나비",
    #     12: "벼룩잎벌레",
    #     13: "복숭아혹진딧물",
    #     14: "비단노린재",
    #     15: "썩덩나무노린재",
    #     16: "알락수염노린재",
    #     17: "열대거세미나방",
    #     18: "큰28점박이무당벌레",
    #     19: "톱다리개미허리노린재",
    #     20: "파밤나방",
    # }
    #
    # predicted_labels = [label_mapping[pred] for pred in preds]
    # original_labels = [label_mapping[actual] for actual in actuals]
    # df = pd.DataFrame({'pred': predicted_labels, 'label': original_labels, 'path': paths})
    df = pd.DataFrame({'pred': preds, 'label': actuals, 'path': paths})

    return df

def check_test(result):
    df = result

    label_list = df['label'].unique().tolist()
    cf = confusion_matrix(df['label'], df['pred'], labels=label_list)
    print(pd.DataFrame(cf, index=label_list, columns=label_list))
    # print(classification_report(df['label'], df['pred'], target_names=label_list))

def check_wrong(result, check_value):
    df = result

    wrongs = df[(df['label'] != df['pred']) & (df['label'] == check_value)]['path'].tolist()

    fig, axes = plt.subplots(1, len(wrongs), figsize=(12, 4))

    for ax, wrong in zip(axes, wrongs):
        ax.imshow(plt.imread(wrong))
        ax.axis('off')

def main():

    test = pd.read_csv("../Output/temp/test_data.csv")

    test_loader = make_loader(test, batch_size=CFG['BATCH_SIZE'], shuffle=False)

    num_classes = test['label'].unique().max() + 1

    vit_model = VITModel(num_classes)
    vit_model.load_state_dict(torch.load("../Output/vit/vit_33.pth"))
    cnn_model = CNNModel(num_classes)
    cnn_model.load_state_dict(torch.load('../Output/cnn/cnn_31.pth'))

    # test
    vit_preds, vit_actuals, vit_paths = test_fn(vit_model, test_loader)
    vit_result = result_df(vit_preds, vit_actuals, vit_paths)
    check_test(vit_result)
    print(vit_result)

    cnn_preds, cnn_actuals, cnn_paths = test_fn(cnn_model, test_loader)
    cnn_result = result_df(cnn_preds, cnn_actuals, cnn_paths)
    check_test(cnn_result)
    print(cnn_result)


if __name__ == '__main__':
    main()