import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
	AutoModelForSequenceClassification,
	BertTokenizer,
	Trainer,
	TrainingArguments,
)

from ..share import DataLoader, DeviceManager

# 定義數據集類別


class SentimentDataset(Dataset):
	def __init__(self, texts, labels, tokenizer, max_length):
		self.texts = texts
		self.labels = labels
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, item):
		text = self.texts[item]
		label = self.labels[item]

		encoding = self.tokenizer.encode_plus(
			text,
			add_special_tokens=True,  # Add [CLS] and [SEP] tokens
			max_length=self.max_length,
			padding="max_length",  # Pad to max length
			truncation=True,  # 文字超出長度是否截斷
			return_attention_mask=True,  # Return attention mask
			return_tensors="pt",  # Return PyTorch tensors
		)

		return {
			"input_ids": encoding["input_ids"].flatten(),
			"attention_mask": encoding["attention_mask"].flatten(),
			"labels": torch.tensor(label, dtype=torch.long),
		}


class SentimentTrainer:
	def __init__(
		self,
		output_dir: str,
		logging_dir: str,
		num_train_epochs: int = 3,
		train_batch_size: int = 4,
		save_steps: int = 100,
		save_total_limit: int = 2,
		logging_steps: int = 10,
		load_best_model_at_end: bool = False,
		warmup_steps: int = 100,
		weight_decay: float = 0.01,
	):
		# 設置訓練參數
		self.training_args = TrainingArguments(
			output_dir=output_dir,  # 訓練結果保存目錄
			num_train_epochs=num_train_epochs,  # 訓練輪數
			per_device_train_batch_size=train_batch_size,  # 訓練批次大小
			save_steps=save_steps,  # 保存檢查點
			save_total_limit=save_total_limit,  # 最多保留兩個檢查點
			logging_dir=logging_dir,  # 日誌保存目錄
			logging_steps=logging_steps,  # 每隔多少步保存日誌
			load_best_model_at_end=load_best_model_at_end,  # 訓練結束後載入最佳模型  # noqa: E501
			warmup_steps=warmup_steps,  # 預熱步數
			weight_decay=weight_decay,  # 權重衰減
		)

	def _check_first_training(self) -> bool:
		# 檢查是否存在訓練檢查點資料夾
		try:
			paths = os.listdir(self.training_args.output_dir)
			checkpoints = [f for f in paths if f.startswith("checkpoint-")]

			# 如果找不到任何檢查點，則是第一次訓練
			if not checkpoints:
				return False

			return True
		except FileNotFoundError:
			return False

	def train_model(self, model, train_dataset):
		# 初始化 Trainer
		trainer = Trainer(
			model=model,  # 使用的模型
			args=self.training_args,  # 訓練參數
			train_dataset=train_dataset,  # 訓練數據集
		)

		# 訓練模型，檢查是否為第一次訓練
		if self._check_first_training():
			trainer.train(resume_from_checkpoint=True)
		else:
			trainer.train(resume_from_checkpoint=False)

		return trainer

	@staticmethod
	def save_model(trainer, model_path):
		# 儲存訓練好的模型
		trainer.save_model(model_path, weights_only=True)


# 載入 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained(
	"nlptown/bert-base-multilingual-uncased-sentiment"
)

if __name__ == "__main__":
	model = AutoModelForSequenceClassification.from_pretrained(
		"nlptown/bert-base-multilingual-uncased-sentiment",
		num_labels=5,
		torch_dtype=torch.float32,
	)

	# 獲取當前腳本目錄
	script_dir = Path(__file__).resolve().parent.parent.parent.parent

	# 拼接相對路徑
	input_file = (
		script_dir / "data" / "custom_data" / "train.json"
	)  # 原始 JSON 文件

	# 檢查設備並設置批次大小
	device_type = DeviceManager.check_device()
	train_batch_size = DeviceManager.setting_batch_size(device_type)
	print("訓練批次大小:", train_batch_size)

	# 訓練模型
	sentiment_trainer = SentimentTrainer(
		output_dir=script_dir / "service/model/results",
		logging_dir=script_dir / "service/model/logs",
		train_batch_size=train_batch_size,
	)

	# 讀取數據並生成批次
	train_texts = []
	train_labels = []
	for batch_texts, batch_labels in DataLoader.read_data_in_batches(
		input_file
	):
		train_texts.extend(batch_texts)
		train_labels.extend(batch_labels)

		print(f"已處理 {len(train_texts)} 條數據")

		train_dataset = SentimentDataset(
			train_texts, train_labels, tokenizer, max_length=128
		)

		trainer_instance = sentiment_trainer.train_model(model, train_dataset)

	# 儲存訓練好的模型
	sentiment_trainer.save_model(
		trainer_instance,
		script_dir / "service/model/sentiment_model",
	)

	# # 評估模型
	# # 在此範例中，假設有測試數據集
	# test_texts = ["這是很棒的經驗", "非常糟糕的服務"]
	# test_labels = [4, 0]
	# test_dataset = SentimentDataset(
	#     test_texts, test_labels, tokenizer, max_length=128
	# )

	# # 設定評估參數並進行評估
	# trainer.evaluate(test_dataset)
