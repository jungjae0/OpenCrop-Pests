import os
import pandas as pd
from sklearn.model_selection import train_test_split


def concat_csv_files(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv') and file.startswith('output')]

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return None

    dfs = []

    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_path)
        dfs.append(df)

    concatenated_df = pd.concat(dfs, ignore_index=True)
    return concatenated_df

def augment_data(aug_dir, raw_data):
    file_list = os.listdir(aug_dir)

    aug_files = pd.DataFrame(file_list, columns=['aug_file'])
    aug_files['raw_file'] = aug_files['aug_file'].str.split(".").str[0].apply(lambda x: x[:-6]) + '.jpg'
    aug_files['aug_path'] = '[T원천]11.토마토/9.증강/' + aug_files['aug_file']

    raw_data = raw_data[raw_data['label'] != 0]

    aug_data = pd.merge(aug_files, raw_data, on='raw_file', how='right')
    aug_data = aug_data[['aug_path', 'label']]

    return aug_data


def split_data(data):
    # 특성과 타겟 분리
    features = data.drop('label', axis=1)
    target = data['label']

    train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2,
                                                                                random_state=42)
    train_data = pd.concat([train_features, train_target], axis=1)
    train_data.to_csv('../Output/train_data.csv', index=False)

    test_data = pd.concat([test_features, test_target], axis=1)
    test_data.to_csv('../Output/test_data.csv', index=False)

def main():
    aug_dir = 'Z:\DATA\노지 작물 해충 진단 이미지\Training\[T원천]11.토마토\9.증강'
    output_dir = '../Output'  # Replace with the path to your output folder
    raw_data = concat_csv_files(output_dir)
    print(raw_data['Class Value'].nunique())

    raw_data = raw_data[(raw_data['Crop'] == 11)]
    raw_data['class'] = raw_data['Folder'].str.split('_').str[1] # 정상, 해충, 충해
    raw_data['root'] = '[T원천]' + raw_data['Folder'].str.split('_').str[0].str.split(']').str[1] # 08.오이, 11.토마토, 05.배추
    raw_data['raw_path'] = raw_data['root'] + '/' + raw_data['class'] + '/' + raw_data['Image Filename']
    raw_data['label'] = raw_data['Class Value'].copy().astype(int)
    raw_data['raw_file'] = raw_data['Image Filename'].copy()

    raw_data = raw_data[['raw_path', 'raw_file', 'label']]
    aug_data = augment_data(aug_dir, raw_data)


    raw = raw_data[['raw_path', 'label']]
    raw.columns = ['path', 'label']
    aug = aug_data
    aug.columns = ['path', 'label']
    data = pd.concat([raw, aug])


    # train/test 데이터 생성
    split_data(data)

if __name__ == '__main__':
    main()
