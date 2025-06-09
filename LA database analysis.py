import pandas as pd
import numpy as np
import os

def matrix_factorization_pipeline(input_excel_path='LA.xlsx', latent_factors=1, learning_rate=0.01, lambda_reg=0.1, epochs=5000):
    """
    执行完整的矩阵分解流程,用于填充Excel表格中的缺失值。

    Args:
        input_excel_path (str): 输入的Excel文件路径。
        latent_factors (int): 潜在因子的数量 (k)。根据您的要求,这里固定为1。
        learning_rate (float): 梯度下降的学习率 。
        lambda_reg (float): 正则化参数 (λ)。
        epochs (int): 迭代训练的次数。

    Returns:
        None: 函数会直接生成三个Excel输出文件。
    """
    # --- 准备工作：如果示例文件不存在，则创建一个 ---
    if not os.path.exists(input_excel_path):
        print(f"未找到 '{input_excel_path}'。正在创建一个 51x20 的示例文件...")
        # 创建一个 51x20 的随机矩阵
        example_data = np.random.rand(51, 20) * 4 + 1 
        # 随机设置约80%的元素为空值 (NaN)
        mask = np.random.rand(51, 20) > 0.2
        example_data[mask] = np.nan
        pd.DataFrame(example_data).to_excel(input_excel_path, index=False, header=False)
        print("示例文件创建成功。您可以替换它为您自己的数据。")

    # 1. 构建原始评分矩阵 R
    print("步骤 1: 正在从Excel读取原始评分矩阵 R...")
    try:
        R = pd.read_excel(input_excel_path, header=None).values.astype(float)
    except FileNotFoundError:
        print(f"错误：输入文件 '{input_excel_path}' 未找到。请确保文件名正确且文件在同一目录下。")
        return

    n_users, n_items = R.shape
    print(f"矩阵维度: {n_users} 个用户, {n_items} 个物品。")

    # 2. 计算全局平均值和偏置项
    print("步骤 2: 正在计算全局平均值 mu 和偏置项 b_i, b_j...")
    mu = np.nanmean(R)

    # 计算用户偏置 b_i
    b_i = np.nanmean(R, axis=1) - mu
    b_i = np.nan_to_num(b_i).reshape(-1, 1) # 处理没有评分的用户并保证维度正确

    # 计算物品偏置 b_j
    b_j = np.nanmean(R, axis=0) - mu
    b_j = np.nan_to_num(b_j).reshape(1, -1) # 处理没有评分的物品并保证维度正确

    # 3. 标准化评分矩阵
    print("步骤 3: 正在标准化矩阵，生成 R'...")
    baseline = mu + b_i + b_j
    R_prime = R - baseline

    # --- 输出 1: 标准化后的矩阵 ---
    df_r_prime = pd.DataFrame(R_prime)
    df_r_prime.to_excel("output_1_normalized_matrix.xlsx", index=False, header=False)
    print("已保存输出 1: 标准化矩阵 (output_1_normalized_matrix.xlsx)")

    # 4. PQ迭代分解
    print(f"步骤 4: 正在进行PQ迭代分解... (epochs={epochs}, lr={learning_rate}, lambda={lambda_reg})")

    P = np.full((n_users, latent_factors), -0.1)
    P[::2] = 0.1 # 奇数行 (索引0, 2, ...) 设置为 0.1

    Q = np.full((n_items, latent_factors), -0.1)
    Q[::2] = 0.1 # 奇数行 (索引0, 2, ...) 设置为 0.1

    non_nan_indices = np.argwhere(~np.isnan(R))

    # 迭代训练
    for epoch in range(epochs):
        np.random.shuffle(non_nan_indices) # 随机打乱顺序
        for u, i in non_nan_indices:
            r_hat_prime = P[u, :] @ Q[i, :].T
            error = R[u, i] - (baseline[u, i] + r_hat_prime)

            p_grad = error * Q[i, :] - lambda_reg * P[u, :]
            q_grad = error * P[u, :] - lambda_reg * Q[i, :]

            P[u, :] += learning_rate * p_grad
            Q[i, :] += learning_rate * q_grad

        if (epoch + 1) % 10 == 0:
            print(f"  ...已完成第 {epoch + 1}/{epochs} 次迭代")

    # 5. 加回偏置项，获得最终预测矩阵
    print("步骤 5: 正在加回偏置项，生成最终预测矩阵 R_hat...")
    R_hat = baseline + (P @ Q.T)

    # --- 输出 2: 最终预测矩阵 ---
    df_r_hat = pd.DataFrame(R_hat)
    df_r_hat.to_excel("output_2_final_predicted_matrix.xlsx", index=False, header=False)
    print("已保存输出 2: 最终预测矩阵 (output_2_final_predicted_matrix.xlsx)")

    # 6. 对最终矩阵进行SVD分解
    print("步骤 6: 正在对最终矩阵进行 SVD 分解...")
    U, s, Vt = np.linalg.svd(R_hat, full_matrices=False)

    S = np.diag(s)

    df_U = pd.DataFrame(U, index=[f'U_row_{i}' for i in range(U.shape[0])])
    df_S = pd.DataFrame(S, index=[f'S_row_{i}' for i in range(S.shape[0])])
    df_Vt = pd.DataFrame(Vt, index=[f'Vt_row_{i}' for i in range(Vt.shape[0])])

    with pd.ExcelWriter("output_3_svd_decomposition.xlsx") as writer:
        df_U.to_excel(writer, sheet_name="U_Matrix")
        df_S.to_excel(writer, sheet_name="S_Matrix")
        df_Vt.to_excel(writer, sheet_name="Vt_Matrix")

    print("已保存输出 3: SVD分解结果 (output_3_svd_decomposition.xlsx),U, S, Vt在不同的工作表中。")
    print("\n所有任务完成!")


if __name__ == '__main__':
    # --- 参数配置 ---
    # 根据您的描述，P是51*1，Q是20*1，这意味着潜在因子数量 k=1
    LATENT_FACTORS = 1 
    
    # 您选择的正则化参数 lambda
    LAMBDA_REG = 0.1
    
    # 其他超参数，可以根据需要调整
    LEARNING_RATE = 0.1 # 学习率
    EPOCHS = 5000 # 迭代次数

    # 执行主函数
    matrix_factorization_pipeline(
        latent_factors=LATENT_FACTORS,
        learning_rate=LEARNING_RATE,
        lambda_reg=LAMBDA_REG,
        epochs=EPOCHS
    )