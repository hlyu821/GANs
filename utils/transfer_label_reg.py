import pandas as pd

# 读取生成的 dataTrain_reg.csv 文件
train_df = pd.read_csv('data12/dataTrain_reg.csv')

# 对 label 列进行分类
unique_labels = train_df['label'].unique()
label_to_class = {label: idx for idx, label in enumerate(sorted(unique_labels))}

# 添加 classlabel 列
train_df['classlabel'] = train_df['label'].map(label_to_class)

# 保存更新后的数据
train_df.to_csv('data12/dataTrain_reg.csv', index=False)