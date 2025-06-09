import pandas as pd
import os

def fill_missing_with_zeros(input_path, output_path=None):
    """
    读取 Excel 表格，填充缺失值为 0，并保存为新的 Excel 文件。
    """
    # 检查文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到文件：{input_path}")
    
    # 读取 Excel 文件
    df = pd.read_excel(input_path, engine='openpyxl')

    # 使用 0 填补空值（包括 NaN）
    df_filled = df.fillna(0)

    # 如果没有指定输出路径，则自动生成一个带 '_filled' 后缀的文件
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_filled{ext}"

    # 保存补零后的表格
    df_filled.to_excel(output_path, index=False, engine='openpyxl')
    print(f"补零后的表格已保存到：{output_path}")


# ✅ 正确调用方式（在函数外部提供路径）
input_path = r"C:\Users\HUAWEI\Desktop\25春\线代\project\normalized_LA.xlsx"
output_path = r"C:\Users\HUAWEI\Desktop\25春\线代\project\normalized_LA_filled.xlsx"

fill_missing_with_zeros(input_path, output_path)
