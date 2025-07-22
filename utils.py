# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

from typing import List, Optional
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import statistics


def fetch_string(raw_str):
    raw_str = raw_str.strip()
    if "```" in raw_str:
        pattern = r"(?s)(?:```json|```)\n([\s\S]*?)\n```"
        match = re.search(pattern, raw_str)
        if match:
            extracted_json = match.group(1)
        else:
            extracted_json = raw_str.replace("```json", "").replace("```", "")
    else:
        extracted_json = raw_str
    return extracted_json


def calculate_statistics(numbers: List[float]) -> dict:
    """
    计算列表的统计指标
    Args:
        numbers: 输入的数字列表
    Returns:
        dict: 包含各种统计指标的字典
    """
    if not numbers:
        return {"error": "Empty list"}

    # 使用 statistics 和 numpy 计算统计指标
    stats = {
        "count":
        len(numbers),  # 样本数
        "mean":
        statistics.mean(numbers),  # 平均值
        "median":
        statistics.median(numbers),  # 中位数
        "min":
        min(numbers),  # 最小值
        "max":
        max(numbers),  # 最大值
        "mode": (statistics.mode(numbers) if len(set(numbers)) < len(numbers)
                 else "No unique mode"),  # 众数
        "std":
        statistics.stdev(numbers) if len(numbers) > 1 else 0,  # 标准差
        "variance":
        statistics.variance(numbers) if len(numbers) > 1 else 0,  # 方差
        # "sum": sum(numbers),  # 总和
        # "range": max(numbers) - min(numbers),  # 极差
        # # 分位数
        # "q1": np.percentile(numbers, 25),  # 第一四分位数
        # "q3": np.percentile(numbers, 75),  # 第三四分位数
        # "iqr": np.percentile(numbers, 75) - np.percentile(numbers, 25),  # 四分位距
    }

    return stats


def keep_letters(s):
    letters = [c for c in s if c.isalpha()]
    result = "".join(letters)
    return result.lower()


def get_md5(string):
    # 创建 MD5 对象
    md5_hash = hashlib.md5()
    # 将字符串编码为字节（MD5 需要字节输入）
    md5_hash.update(string.encode("utf-8"))
    # 获取 16 进制表示的哈希值
    return md5_hash.hexdigest()


def draw_distrubute(scores, fig_name="autoschorlar_benchmark_pasa_socre_dist"):
    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 方法 1：绘制直方图
    plt.subplot(1, 2, 1)  # 1行2列，第1个子图
    plt.hist(scores, bins=10, range=(0, 1), color="skyblue", edgecolor="black")
    plt.title("Histogram of Scores (0-1)")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.xlim(0, 1)  # 设置 x 轴范围为 0 到 1
    plt.xticks(np.arange(0, 1.1, 0.1))  # 设置 x 轴刻度为 0.1 间隔

    # 方法 2：绘制核密度估计图（KDE）
    plt.subplot(1, 2, 2)  # 1行2列，第2个子图
    sns.kdeplot(scores, color="purple", fill=True)
    plt.title("KDE of Scores (0-1)")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.xlim(0, 1)  # 设置 x 轴范围为 0 到 1
    plt.xticks(np.arange(0, 1.1, 0.1))  # 设置 x 轴刻度为 0.1 间隔

    plt.suptitle(fig_name, fontsize=16, y=1.05)

    # 调整布局，避免重叠
    plt.tight_layout()

    # 保存图形到文件（PNG格式）
    image_name = "_".join(i for i in fig_name.split(" "))
    imgfile = f"./metrics/img/{image_name}.png"
    plt.savefig(imgfile, dpi=300, bbox_inches="tight")
    # 关闭图形，释放内存
    plt.close()
    print(f"image saved to: {imgfile}")

from filelock import FileLock
import pandas as pd
import os

def save_to_excel(df_new: pd.DataFrame, excel_file: str, sheet_name: str) -> None:
    """
    Thread-safe function to save DataFrame to Excel with file locking.

    Args:
        df_new: New data to append
        excel_file: Target Excel file path
        sheet_name: Target sheet name
    """
    # Define fixed columns
    fixed_front_cols = [
        "Model Name","Score Threshold","F1", "Recall After Sim Filter","Precision","Avg Sim Score of Recalled Papers","Recall Raw Doc num mean","Recall After Filter Doc num mean"
    ]
    fixed_last_col = "DESCRIB"

    # Sort columns dynamically
    remaining_cols = [col for col in df_new.columns if col not in fixed_front_cols + [fixed_last_col]]
    recall_cols = sorted([col for col in remaining_cols if "recall" in col.lower()], key=str.lower)
    precision_cols = sorted([col for col in remaining_cols if "precision" in col.lower()], key=str.lower)
    other_cols = sorted([col for col in remaining_cols if col not in recall_cols + precision_cols], key=str.lower)
    final_cols = fixed_front_cols + recall_cols + precision_cols + other_cols + [fixed_last_col]
    final_cols = [col for col in final_cols if col in df_new.columns]
    df_new = df_new[final_cols]

    def write_best_f1(writer: pd.ExcelWriter, df: pd.DataFrame, suffix: str = "_best") -> None:
        if "F1" in df.columns:
            df_best = df.loc[df.groupby("Model Name")["F1"].idxmax()].reset_index(drop=True)
            df_best = df_best.sort_values(by="F1", ascending=False).reset_index(drop=True)
            df_best = df_best[final_cols]
            df_best.to_excel(writer, sheet_name=f"{sheet_name}{suffix}", index=False)
        else:
            print(f"Column 'F1' not found in data, skipping '{sheet_name}{suffix}' creation.")

    # Use file locking to prevent concurrent writes
    lock_file = excel_file + ".lock"
    with FileLock(lock_file, timeout=60):  # Timeout after 60 seconds
        try:
            if os.path.exists(excel_file):
                try:
                    with pd.ExcelFile(excel_file) as xls:
                        if sheet_name in xls.sheet_names:
                            # Read existing data and merge
                            df_existing = pd.read_excel(excel_file, sheet_name=sheet_name)
                            df_updated = pd.concat([df_existing, df_new], ignore_index=True)
                            df_updated = df_updated.drop_duplicates(
                                subset=["Model Name", "Score Threshold"], keep="last").reset_index(drop=True)
                            df_updated = df_updated[final_cols]
                            # Update existing sheet and add _best sheet
                            with pd.ExcelWriter(excel_file, mode="a", if_sheet_exists="replace") as writer:
                                df_updated.to_excel(writer, sheet_name=sheet_name, index=False)
                                write_best_f1(writer, df_updated)
                        else:
                            # Append new sheet
                            with pd.ExcelWriter(excel_file, mode="a") as writer:
                                df_new.to_excel(writer, sheet_name=sheet_name, index=False)
                                write_best_f1(writer, df_new)
                            print(f"Worksheet '{sheet_name}' not found, created new one.")
                except (pd.errors.EmptyDataError, ValueError, Exception) as read_error:
                    print(f"Warning: Existing file may be corrupted ({str(read_error)}). Overwriting.")
                    with pd.ExcelWriter(excel_file, mode="w") as writer:
                        df_new.to_excel(writer, sheet_name=sheet_name, index=False)
                        write_best_f1(writer, df_new)
            else:
                # Create new file
                with pd.ExcelWriter(excel_file, mode="w") as writer:
                    df_new.to_excel(writer, sheet_name=sheet_name, index=False)
                    write_best_f1(writer, df_new)
                print(f"Created new Excel file: {excel_file}")
        except Exception as e:
            raise ValueError(f"Error processing Excel file: {str(e)}")
# Install filelock if needed: pip install filelock