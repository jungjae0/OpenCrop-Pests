import os
import json
import csv
import tqdm
import pandas as pd

def filtering(root_folder):
    dir_list = os.listdir(root_folder)

    filtered_folders = []

    for folder_name in dir_list:
        if '라벨링' in folder_name and ('오이' in folder_name or '토마토' in folder_name or '배추' in folder_name) and not folder_name.endswith('.zip'):
            filtered_folders.append(folder_name)

    return filtered_folders

def generate_csv(root_folder, csv_filename):
    dirs_list = filtering(root_folder)

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write header row
        header = ['Folder', 'Image Filename', 'Crop', 'Class Value', 'Grow']
        csv_writer.writerow(header)

        for folder_name in tqdm.tqdm(dirs_list):
            folder_path = os.path.join(root_folder, folder_name)  # Replace with actual root directory

            for filename in os.listdir(folder_path):
                if filename.endswith('.json'):
                    json_path = os.path.join(folder_path, filename)
                    with open(json_path, 'r', encoding='utf-8') as json_file:
                        data = json.load(json_file)
                        image_filename = data['description']['image']
                        crop = data['annotations']['crop']
                        class_value = data['annotations']['object'][0]['class']
                        grow = data['annotations']['object'][0]['grow']

                        csv_writer.writerow([folder_name, image_filename, crop, class_value, grow])

    print(f"CSV file '{csv_filename}' created successfully.")

def generate_dataset(root_folder):
    dirs_list = filtering(root_folder)

    # Create lists to hold data
    folder_list = []
    image_filename_list = []
    crop_list = []
    class_value_list = []
    grow_list = []

    for folder_name in tqdm.tqdm(dirs_list):
        folder_path = os.path.join(root_folder, folder_name)  # Replace with actual root directory

        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                json_path = os.path.join(folder_path, filename)
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    image_filename = data['description']['image']
                    crop = data['annotations']['crop']
                    class_value = data['annotations']['object'][0]['class']
                    grow = data['annotations']['object'][0]['grow']

                    folder_list.append(folder_name)
                    image_filename_list.append(image_filename)
                    crop_list.append(crop)
                    class_value_list.append(class_value)
                    grow_list.append(grow)

    # Create a dictionary to hold the lists
    data_dict = {
        'Folder': folder_list,
        'Image Filename': image_filename_list,
        'Crop': crop_list,
        'Class Value': class_value_list,
        'Grow': grow_list
    }

    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data_dict)

    return df

def main():
    root_folder = r"Z:\DATA\노지 작물 해충 진단 이미지\Training"


    output_dir = '../Output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # csv_filename = os.path.join(output_dir, 'output.csv')

    df = generate_dataset(root_folder)



if __name__ == '__main__':
    main()