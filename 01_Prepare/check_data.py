import pandas as pd


def check_len(crop):
    data = pd.read_csv('../Output/concatenated_data.csv')

    total = data[data['path'].str.contains(f'{crop}')]
    categories = ['정상', '해충', '충해']

    results = []

    for category in categories:
        filtered = total[total['path'].str.contains(category)]
        results.append(len(filtered))

    df = pd.DataFrame({
        'crop': [crop],
        '정상': [results[0]],
        '해충': [results[1]],
        '충해': [results[2]],
        '전체': [len(total)]
    })

    return df


crops = ["오이", "배추", "토마토"]
all_dfs = []

for crop in crops:
    crop_df = check_len(crop)
    all_dfs.append(crop_df)

# 모든 데이터프레임을 하나로 합칩니다.
final_df = pd.concat(all_dfs, ignore_index=True)

print(final_df)

test = pd.read_csv('../Output/temp/test_data.csv')
train = pd.read_csv('../Output/temp/train_data.csv')


testcounts = test['label'].value_counts()
traincounts = train['label'].value_counts()


print("test")
print(testcounts)
print("train")
print(traincounts)
