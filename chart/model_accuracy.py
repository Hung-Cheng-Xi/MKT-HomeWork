import matplotlib.pyplot as plt
import os

# 嵌入的模型名稱與精準度
model_names = [
    "sentiment_model",
    "sentiment_model_25",
    "sentiment_model_50",
    "sentiment_model_75",
    "kv_fold_results_25",
]

accuracies = [0.6880, 0.8910, 0.8730, 0.8570, 0.7070]

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

current_dir = os.path.dirname(os.path.abspath(__file__))  # 取得執行檔案所在的目錄
output_path = os.path.join(current_dir, "model_accuracy.png")
plt.savefig(output_path)  # 儲存圖片到指定路徑

plt.show()
