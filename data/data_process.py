import json
import random
from pathlib import Path

# 獲取當前腳本目錄
script_dir = Path(__file__).resolve().parent

# 拼接相對路徑
input_file = (
    script_dir / "source_data" / "yelp_academic_dataset_review.json"
)  # 原始 JSON 文件
train_file = script_dir / "custom_data" / "train.json"  # 訓練數據文件
test_file = script_dir / "custom_data" / "test.json"  # 測試數據文件

# 設置隨機種子，確保可重現性
random.seed(42)

# 初始化容器
train_data = []
test_data = []

# 分割比例
train_ratio = 0.8

print("開始讀取原始數據文件...")

# 逐行讀取大文件
with open(input_file, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        # 解析每一行的 JSON
        review = json.loads(line)

        # 提取所需字段
        filtered_review = {
            "review_id": review["review_id"],
            "business_id": review["business_id"],
            "stars": review["stars"],
            "text": review["text"],
        }

        # 隨機分配到訓練或測試集
        if random.random() < train_ratio:
            train_data.append(filtered_review)
        else:
            test_data.append(filtered_review)

        # 每讀取 10000 行，打印一次進度
        if (i + 1) % 10000 == 0:
            print(f"已讀取 {i + 1} 行")

print("原始數據文件讀取完畢")
print(f"總共讀取了 {i + 1} 行")

print("開始儲存訓練數據文件...")
# 儲存到文件
with open(train_file, "w", encoding="utf-8") as train_out:
    for review in train_data:
        train_out.write(json.dumps(review) + "\n")
print("訓練數據文件儲存完畢")

print("開始儲存測試數據文件...")
with open(test_file, "w", encoding="utf-8") as test_out:
    for review in test_data:
        test_out.write(json.dumps(review) + "\n")
print("測試數據文件儲存完畢")

print(f"訓練數據: {len(train_data)} 條")
print(f"測試數據: {len(test_data)} 條")
