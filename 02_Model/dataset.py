import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class CustomImageDataset(Dataset):
    def __init__(self, data, transforms=None, cache=None):
        self.data = data.copy()
        self.transforms = transforms
        self.num_classes = 17
        self.cache_image = cache
        # self.imgdir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # img_path = self.data.iloc[idx, 0]
        # img_path = os.path.join('Z:\DATA\노지 작물 해충 진단 이미지\Training', img_path)
        # img_array = np.fromfile(img_path, np.uint8) # 한글 경로 문제 해결: 바이너리 데이터를 넘파이 행렬로 읽고 복호화
        #
        # image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        #
        # image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
        #
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image /= 255.
        #
        # if self.transforms is not None:
        #     image = self.transforms(image = image)['image']
        #
        # label = self.data.iloc[idx, 1]
        #
        # image = torch.as_tensor(image, dtype=torch.float32)
        # label = torch.as_tensor(label, dtype=torch.long)


        def get_image_cache(idx):
            img_path = self.data.iloc[idx, 0]
            img_path = os.path.join('D:\DATA\노지 작물 해충 진단 이미지\Training', img_path)
            if img_path in self.cache_image:
                return img_path, self.cache_image[img_path]
            else:
                img_array = np.fromfile(img_path, np.uint8)  # 한글 경로 문제 해결: 바이너리 데이터를 넘파이 행렬로 읽고 복호화

                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                image = cv2.cvtColor(image, cv2.IMREAD_COLOR)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
                image /= 255.

                if self.transforms is not None:
                    image = self.transforms(image=image)['image']

                self.cache_image[img_path] = torch.as_tensor(image, dtype=torch.float32)
                return img_path, self.cache_image[img_path]

        img_path, image = get_image_cache(idx)
        label = self.data.iloc[idx, 1]

        return {'image': image, 'label': label, 'path': img_path}


def transform():
    return A.Compose([
        A.Resize(224,224),
        ToTensorV2()])

def make_loader(data, batch_size, shuffle, cache):
    return DataLoader(
        dataset=CustomImageDataset(data, transforms=transform(), cache=cache),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=7,
    )
