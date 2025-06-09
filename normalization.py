import pandas as pd
import numpy as np

# 设置文件路径
input_path = r"C:\Users\HUAWEI\Desktop\25春\线代\project\LA.xlsx"
output_path = r"C:\Users\HUAWEI\Desktop\25春\线代\project\normalized_LA.xlsx"

# 读取 Excel 文件
df = pd.read_excel(input_path)

# 提取用户 ID 和歌曲名称
user_ids = df.iloc[:, 0]            # 第一列：用户编号
song_names = df.columns[1:]         # 第一行（去掉第一列）：歌曲名称

# 提取评分矩阵
R = df.iloc[:, 1:].values           # 只要评分部分

# 创建掩码（True 表示有评分）
mask = R > 0
mu = R[mask].mean()                 # 全局平均评分（仅限非 0 项）

# 用户偏差 b_i
b_i = np.zeros(R.shape[0])
for i in range(R.shape[0]):
    rated = mask[i, :]
    if rated.sum() > 0:
        b_i[i] = R[i, rated].mean() - mu

# 歌曲偏差 b_j
b_j = np.zeros(R.shape[1])
for j in range(R.shape[1]):
    rated = mask[:, j]
    if rated.sum() > 0:
        b_j[j] = R[rated, j].mean() - mu

# 构建归一化评分矩阵
R_prime = np.full_like(R, fill_value=np.nan, dtype=float)
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        if mask[i, j]:
            R_prime[i, j] = R[i, j] - mu - b_i[i] - b_j[j]

# 转换为 DataFrame，保留原用户 ID 和列名
df_prime = pd.DataFrame(R_prime, columns=song_names)
df_prime.insert(0, 'User', user_ids)

# 保存结果到 Excel 文件
df_prime.to_excel(output_path, index=False)

print(f"归一化完成，结果已保存为：{output_path}")

