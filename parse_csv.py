import pandas
input_csv = ".\\image_path.tsv"

df = pandas.read_csv(input_csv)

for idx, row in df.iterrows():
    df.loc[idx, 'img_path'] = "C:\\dev\\mri" + df.loc[idx, 'img_path'].lstrip('.')
    # print (df.loc[idx, 'img_path'])
    if (idx % 500 == 0):
        print ('Finish %d.' % idx)
df.to_csv(".\\img_path.csv", sep=",")


