from pathlib import Path
from typing import Dict

from matplotlib import pyplot as plt
from collections import Counter


def plot_top_foods_pie_chart(data: Dict) -> None:
	# 計算食物出現次數
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

	# 獲取當前腳本目錄
	script_dir = Path(__file__).resolve().parent.parent

	# 拼接相對路徑
	picture_file = script_dir / "assets" / "top_foods_pie_chart.png"

	plt.savefig(picture_file)  # 儲存圖片到指定路徑

	plt.show()