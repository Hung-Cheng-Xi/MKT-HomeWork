from pathlib import Path
from sklearn.metrics import accuracy_score
from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	pipeline,
)
from ..share import DataLoader
import torch


# 設定檔案路徑與模型列表
# 獲取當前腳本目錄
script_dir = Path(__file__).resolve().parent.parent.parent.parent

# 拼接相對路徑
input_file = (
	script_dir / "data" / "custom_data" / "test.json"
)  # 原始 JSON 文件

model_names = [
	"service/model/sentiment_model",
	"service/model/sentiment_model_25",
	"service/model/sentiment_model_50",
	"service/model/sentiment_model_75",
	"service/model/kv_fold_results_25",
]

# 儲存結果
results = {}

for model_name in model_names:
	print(f"Evaluating model: {model_name}")

	# 加載模型與分詞器
	tokenizer = AutoTokenizer.from_pretrained(
		"nlptown/bert-base-multilingual-uncased-sentiment"
	)
	model = AutoModelForSequenceClassification.from_pretrained(
		model_name, num_labels=5
	)
	# model.to(device)

	print("模型載入完成")

	# 創建推論管道
	classifier = pipeline(
		"text-classification",
		model=model,
		tokenizer=tokenizer,
		device=0 if torch.cuda.is_available() else -1,
	)

	print("推論管道創建完成")

	# 預測與計算準確度
	all_preds = []
	all_labels = []

	# 讀取數據並生成批次
	for batch_texts, batch_labels in DataLoader.read_data_in_batches(
		input_file
	):
		# 預測
		batch_texts = [text[:512] for text in batch_texts]
		preds = classifier(batch_texts)
		predicted_labels = [
			int(pred["label"].strip().split()[0]) - 1 for pred in preds
		]

		all_preds.extend(predicted_labels)
		all_labels.extend(batch_labels)

		print(f"已處理 {len(all_labels)} 條數據")
		break

	# 計算準確度
	accuracy = accuracy_score(all_labels, all_preds)
	results[model_name] = accuracy
	print(f"Accuracy for {model_name}: {accuracy:.4f}")
	print("-" * 50)

# 輸出結果
print("\nFinal Results:")
for model_name, acc in results.items():
	print(f"{model_name}: {acc:.4f}")
