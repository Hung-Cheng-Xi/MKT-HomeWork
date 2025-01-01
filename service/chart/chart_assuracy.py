from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt


def plot_model_accuracy(data: Dict[str, float]) -> None:
	"""
	繪製模型精準度比較圖表，並儲存圖片到指定路徑，並顯示圖表。
	data: 嵌入的模型名稱與精準度
	"""

	# 嵌入的模型名稱與精準度
	model_names, accuracies = zip(*data.items())

	# 繪圖
	plt.figure(figsize=(15, 6))
	plt.bar(model_names, accuracies, color="skyblue")
	plt.title("Model Accuracy Comparison", fontsize=16)
	plt.xlabel("Model", fontsize=14)
	plt.ylabel("Accuracy", fontsize=14)
	plt.xticks(rotation=0, ha="center", fontsize=12)  # 文字水平顯示
	plt.ylim(0, 1)  # 精準度範圍 0~1
	plt.grid(axis="y", linestyle="--", alpha=0.7)

	# 顯示數值標籤
	for i, acc in enumerate(accuracies):
		plt.text(i, acc + 0.01, f"{acc:.4f}", ha="center", fontsize=12)

	plt.tight_layout()

	# 設定檔案路徑與模型列表
	# 獲取當前腳本目錄
	script_dir = Path(__file__).resolve().parent.parent

	# 拼接相對路徑
	picture_file = (
		script_dir / "assets" / "model_accuracy.png"
	)  # 原始 JSON 文件

	plt.savefig(picture_file)  # 儲存圖片到指定路徑

	plt.show()
