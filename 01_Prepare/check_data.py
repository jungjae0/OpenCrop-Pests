import pandas as pd
import os

data_dir = "../Output"

raw_train = pd.read_csv(os.path.join(data_dir, "raw_train_data.csv"))
raw_test = pd.read_csv(os.path.join(data_dir, "raw_test_data.csv"))
aug_train = pd.read_csv(os.path.join(data_dir, "aug_train_data.csv"))
aug_test = pd.read_csv(os.path.join(data_dir, "aug_test_data.csv"))

data_list = [raw_train, raw_test, aug_train, aug_test]

# raw_train_counts = raw_train['label'].value_counts()
# raw_test_counts = raw_test['label'].value_counts()
# aug_train_counts = aug_train['label'].value_counts()
# aug_test_counts = aug_test['label'].value_counts()

value_counts_list = []
data_len_list = []
for data in data_list:
    value_counts_list.append(data['label'].value_counts())
    data_len_list.append(len(data))


info = pd.concat(value_counts_list, axis=1)
info.columns = ['raw_train', 'raw_test', 'aug_train', 'aug_test']
info.loc[5] = data_len_list
info.index = ['정상', '알락수염노린재', '담배가루이', '꽃노랑총채벌레', '비단노린재', '데이터수']


print(info)



