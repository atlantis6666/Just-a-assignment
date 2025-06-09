import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ======= Step 0: 文件路径（请确保路径无误）=======
rating_path = "C:/Users/HUAWEI/Desktop/25春/线代/project/normalized_LA_filled.xlsx"
tag_path = "C:/Users/HUAWEI/Desktop/25春/线代/project/tag_final.xlsx"

# ======= Step 1: 读取评分矩阵 R =======
rating_df = pd.read_excel(rating_path, header=0)
user_ids = rating_df.iloc[:, 0]  # 用户编号
song_names = rating_df.columns[1:]  # 歌曲名（跳过第一列）
R = rating_df.iloc[:, 1:].values  # 评分矩阵 (num_users x num_songs)

# ======= Step 2: 读取 tag 向量矩阵 V（歌曲特征）=======
tag_df = pd.read_excel(tag_path, header=0, index_col=0)
V = tag_df.values.astype(float)  # V 是 (num_tags x num_songs)

# ======= Step 3: 计算每个用户的偏好向量 u ∈ R^num_tags =======
num_users = R.shape[0]
num_tags, num_songs = V.shape

U = np.zeros((num_users, num_tags))

for i in range(num_users):
    ratings = R[i, :]
    mask = ratings != 0
    if np.sum(mask) == 0:
        continue  # 如果该用户没有评分则跳过
    weighted_sum = np.dot(V[:, mask], ratings[mask])  # (12 x n) × (n,) = (12,)
    total_weight = np.sum(ratings[mask])
    U[i, :] = weighted_sum / total_weight  # 加权平均

# ======= Step 4: 计算 Cosine Similarity：每个用户与每首歌 =======
similarity_matrix = cosine_similarity(U, V.T)  # 结果为 (num_users x num_songs)

# ======= Step 5: 写入 Excel 输出 =======

# 用户偏好向量表格
U_df = pd.DataFrame(U, columns=tag_df.index)
U_df.insert(0, "User ID", user_ids)
U_df.to_excel("user_preference_vectors.xlsx", index=False)

# 相似度表格
sim_df = pd.DataFrame(similarity_matrix, columns=song_names)
sim_df.insert(0, "User ID", user_ids)
sim_df.to_excel("similarity_matrix.xlsx", index=False)

print("✅ 输出完成：")
print("user_preference_vectors.xlsx：用户的 tag 偏好向量")
print("similarity_matrix.xlsx：用户对每首歌的余弦相似度")
