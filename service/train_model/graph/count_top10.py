from pathlib import Path
from typing import Dict
from matplotlib import pyplot as plt
from collections import Counter
import json


with open('data/custom_data/output.json', 'r', encoding='utf-8') as f:
   data = json.load(f)
food_counter = Counter(item["food"] for item in data)
print("data", data, "food_counter", len(food_counter))
top_foods = food_counter.most_common(10)
# 提取食物名稱與出現次數
food_names, counts = zip(*top_foods)
# 繪製圓餅圖
plt.figure(figsize=(10, 6))
plt.pie(
	counts,
	labels=food_names,
	autopct="%1.1f%%",
	startangle=140,
	colors=plt.cm.Paired.colors,
)
plt.title("Top 10 Foods by Frequency", fontsize=16)
plt.axis("equal")  # 保持圓餅圖為圓形
# 拼接相對路徑
plt.savefig('data/custom_data/count_top10.png')
plt.show()