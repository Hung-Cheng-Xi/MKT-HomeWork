import json
from pathlib import Path

# 獲取當前腳本目錄
script_dir = Path(__file__).resolve().parent

# 拼接相對路徑
input_file = (
	script_dir / "source_data" / "yelp_academic_dataset_review.json"
)  # 原始 JSON 文件
output_250k_file = (
	script_dir / "custom_data" / "data_250k.json"
)  # 25萬筆數據文件
output_500k_file = (
	script_dir / "custom_data" / "data_500k.json"
)  # 50萬筆數據文件
output_750k_file = (
	script_dir / "custom_data" / "data_750k.json"
)  # 75萬筆數據文件
output_75k_file = (
	script_dir / "custom_data" / "data_75k.json"
)  # 75萬筆數據文件

# 初始化容器
data_250k = []
data_500k = []
data_750k = []
data_75k = []

# 設定目標數據量
target_counts = [250000, 500000, 750000, 825000]

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

		# 根據行數填充到對應數據容器
		if i < target_counts[0]:
			data_250k.append(filtered_review)
		if i < target_counts[1]:
			data_500k.append(filtered_review)
		if i < target_counts[2]:
			data_750k.append(filtered_review)
		if i >= target_counts[2] and i <= target_counts[3]:
			data_75k.append(filtered_review)

		# 當最大目標數據量達成時停止讀取
		if i + 1 >= target_counts[3]:
			break

		# 每讀取 10000 行，打印一次進度
		if (i + 1) % 10000 == 0:
			print(f"已讀取 {i + 1} 行")

print("原始數據文件讀取完畢")
print(f"總共讀取了 {i + 1} 行")

# 儲存 25萬筆數據
print("開始儲存 25萬筆數據文件...")
with open(output_250k_file, "w", encoding="utf-8") as out_250k:
	for review in data_250k:
		out_250k.write(json.dumps(review) + "\n")
print("25萬筆數據文件儲存完畢")

# 儲存 50萬筆數據
print("開始儲存 50萬筆數據文件...")
with open(output_500k_file, "w", encoding="utf-8") as out_500k:
	for review in data_500k:
		out_500k.write(json.dumps(review) + "\n")
print("50萬筆數據文件儲存完畢")

# 儲存 75萬筆數據
print("開始儲存 75萬筆數據文件...")
with open(output_750k_file, "w", encoding="utf-8") as out_750k:
	for review in data_750k:
		out_750k.write(json.dumps(review) + "\n")
print("75萬筆數據文件儲存完畢")

# 儲存 7.5萬筆數據
print("開始儲存 7.5萬筆數據文件...")
with open(output_75k_file, "w", encoding="utf-8") as out_75k:
	for review in data_75k:
		out_75k.write(json.dumps(review) + "\n")
print("7.5萬筆數據文件儲存完畢")

print(f"25萬筆數據: {len(data_250k)} 條")
print(f"50萬筆數據: {len(data_500k)} 條")
print(f"75萬筆數據: {len(data_750k)} 條")
print(f"7.5萬筆數據: {len(data_75k)} 條")
