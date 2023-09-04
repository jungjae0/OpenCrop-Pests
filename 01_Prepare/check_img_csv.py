import pandas as pd
import os


def raw_img_files(img_dir):


    file_list = []

    categories = ['0.정상', '2.해충', '3.충해']

    for category in categories:
        category_dir = os.path.join(img_dir, category)

        if os.path.exists(category_dir) and os.path.isdir(category_dir):
            img_files = [f for f in os.listdir(category_dir) if f.endswith('.jpg') or f.endswith('.JPG')]
            for img_file in img_files:
                file_list.append({'Image Filename': img_file})


    img_files = pd.DataFrame(file_list)

    return img_files

def aug_img_files(aug_dir):
    file_list = os.listdir(aug_dir)
    file_list = [f for f in file_list if f.endswith('.jpg') or f.endswith('.JPG')]

    aug_files = pd.DataFrame(file_list, columns=['Image Filename'])

    return aug_files

img_dir = r"D:\DATA\노지 작물 해충 진단 이미지\Training\[T원천]11.토마토"
aug_dir = os.path.join(img_dir, '9.증강')

output_dir = '../Output'
raw = pd.read_csv(os.path.join(output_dir, "raw_data.csv"))
aug = pd.read_csv(os.path.join(output_dir, "aug_data.csv"))
all = pd.read_csv(os.path.join(output_dir, "all_data.csv"))


raw_img = raw_img_files(img_dir)
aug_img = aug_img_files(aug_dir)
all_img = pd.concat([raw_img, aug_img])

print(all_img.columns)
print(all.columns)

check = pd.merge(all, all_img, left_on='path', right_on='Image Filename', how='inner')

un_all_img = all_img[~all_img['Image Filename'].isin(check['Image Filename'])]
un_all = all[~all['path'].isin(check['Image Filename'])]

print(len(un_all_img))
print(len(un_all))
print(len(check))
