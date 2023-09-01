import os
import pandas as pd
from sklearn.model_selection import train_test_split

def raw_json_files(output_dir):
    csv_files = ['jsons/output4.csv', 'jsons/output5.csv', 'jsons/output6.csv']

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return None

    dfs = []

    for csv_file in csv_files:
        csv_path = os.path.join(output_dir, csv_file)
        df = pd.read_csv(csv_path)
        dfs.append(df)

    json_files = pd.concat(dfs, ignore_index=True)
    json_files['Folder'] = json_files['Folder'].str.split(']').str[1]
    json_files['Folder'] = '[T원천]' + json_files['Folder']
    json_files = json_files[['Folder', 'Image Filename', 'Class Value']]

    return json_files

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

def aug_img_files(aug_dir, raw_data):
    file_list = os.listdir(aug_dir)
    file_list = [f for f in file_list if f.endswith('.jpg') or f.endswith('.JPG')]

    aug_files = pd.DataFrame(file_list, columns=['aug_file'])

    aug_files['aug_extension'] = aug_files['aug_file'].str.split('.').str[1]
    aug_files['raw_file'] = aug_files['aug_file'].str.split(".").str[0].apply(lambda x: x[:-6])

    aug_files['aug_path'] = '[T원천]11.토마토/9.증강/' + aug_files['aug_file']

    raw_data = raw_data[raw_data['label'] != 0]
    raw_data.loc[:, 'raw_file'] = raw_data['raw_file'].str.split('.').str[0]

    aug_data = pd.merge(aug_files, raw_data, on='raw_file', how='right')
    aug_data = aug_data[['aug_path', 'label']]

    return aug_data


def get_raw_data(img_dir, output_dir):
    img_files = raw_img_files(img_dir)
    json_files = raw_json_files(output_dir)

    raw_data = pd.merge(img_files, json_files, on='Image Filename', how='inner')
    raw_data['raw_path'] = raw_data['Folder'].str.replace("_", "/") + '/' + raw_data['Image Filename']


    raw_data = raw_data[['Image Filename', 'raw_path', 'Class Value']]
    raw_data.columns = ['raw_file', 'raw_path', 'label']

    return raw_data

def split_data(data, filename):
    # 특성과 타겟 분리
    features = data.drop('label', axis=1)
    target = data['label']

    train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2,
                                                                                random_state=42)
    train_data = pd.concat([train_features, train_target], axis=1)
    train_data.to_csv(f'../Output/{filename}_train_data.csv', index=False)

    test_data = pd.concat([test_features, test_target], axis=1)
    test_data.to_csv(f'../Output/{filename}_test_data.csv', index=False)


# def check_data():
#     # aug_imgs
#     aug_imgs = [f for f in os.listdir(aug_dir) if f.endswith('.jpg') or f.endswith('.JPG')]
#     print("전체 증강 이미지: ", len(aug_imgs))
#     print("라벨 부여 수행한 증강 이미지: ", len(aug_data))
#
#     # json_files
#     print("원천 라벨링 데이터: ", len(json_files))
#
#     # raw_imgs
#     print("원천 이미지 데이터: ", len(img_files))
#
#     # total
#     print("이미지 데이터: ", len(img_files) + len(aug_imgs))
#     print("라벨 데이터: ", len(aug_data) + len(raw_data))
#     print("concat 데이터: ", len(data))


def main():
    img_dir = r"Z:\DATA\노지 작물 해충 진단 이미지\Training\[T원천]11.토마토"
    aug_dir = os.path.join(img_dir, '9.증강')

    output_dir = '../Output'

    raw_data = get_raw_data(img_dir, output_dir)
    aug_data = aug_img_files(aug_dir, raw_data)

    raw = raw_data[['raw_path', 'label']]
    raw.columns = ['path', 'label']
    aug = aug_data
    aug.columns = ['path', 'label']
    all_data = pd.concat([raw, aug])

    # train/test 데이터 생성
    split_data(raw, 'raw')
    split_data(all_data, 'aug')

if __name__ == '__main__':
    main()


    # print(len(raw_data))
    # print(len(img_files))
    # print(len(json_files))
    #
    # unmerged_img_data = img_files[~img_files['Image Filename'].isin(raw_data['Image Filename'])]
    # unmerged_json_data = json_files[~json_files['Image Filename'].isin(raw_data['Image Filename'])]
    #
    # print("Unmerged Image Data:")
    # print(unmerged_img_data)
    #
    # print("\nUnmerged JSON Data:")
    # print(unmerged_json_data)
    # unmerged_json_data.to_csv(os.path.join(output_dir, "unmerged_json.csv"), index=False)
    # img_files.to_csv(os.path.join(output_dir, "img_files.csv"), index=False)
    # json_files.to_csv(os.path.join(output_dir, "json_files.csv"), index=False)
    # raw_data.to_csv(os.path.join(output_dir, "raw_data.csv"), index=False)
    #
    #
    #
    # normal = os.listdir(os.path.join(img_dir, "0.정상"))
    # a = os.listdir(os.path.join(img_dir, "2.해충"))
    # b = os.listdir(os.path.join(img_dir, "3.충해"))
    #
    # n_len = [f for f in normal if f.endswith('.jpg') or f.endswith('.JPG')]
    # a_len = [f for f in a if f.endswith('.jpg') or f.endswith('.JPG')]
    # b_len = [f for f in b if f.endswith('.jpg') or f.endswith('.JPG')]
    #
    # print(len(n_len) + len(a_len) + len(b_len))