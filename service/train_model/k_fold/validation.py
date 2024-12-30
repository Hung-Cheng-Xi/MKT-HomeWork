import torch
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
)
from transformers import (
	AutoModelForSequenceClassification,
	BertTokenizer,
	Trainer,
	TrainingArguments,
)

from ..share import (
	DataLoader,
	DeviceManager,
	SentimentDataset,
)


class SentimentTrainer:
	def __init__(
		self,
		output_dir: str,
		logging_dir: str,
		num_train_epochs: int = 1,
		train_batch_size: int = 4,
		save_steps: int = 600,
		save_total_limit: int = 2,
		logging_steps: int = 150,
		load_best_model_at_end: bool = True,
		warmup_steps: int = 450,
		weight_decay: float = 0.01,
		weight_regularization: float = 0.0,
		batch_normalization: bool = False,
	):
		# 設置訓練參數
		self.training_args = TrainingArguments(
			output_dir=output_dir,  # 訓練結果保存目錄
			num_train_epochs=num_train_epochs,  # 訓練輪數
			per_device_train_batch_size=train_batch_size,  # 訓練批次大小
			per_device_eval_batch_size=train_batch_size,  # 評估批次大小
			save_steps=save_steps,  # 保存檢查點
			save_total_limit=save_total_limit,  # 最多保留兩個檢查點
			logging_dir=logging_dir,  # 日誌保存目錄
			logging_steps=logging_steps,  # 每隔多少步保存日誌
			load_best_model_at_end=load_best_model_at_end,  # 訓練結束後載入最佳模型  # noqa: E501
			warmup_steps=warmup_steps,  # 預熱步數
			weight_decay=weight_decay,  # 權重衰減
			fp16=True,
			fp16_opt_level="O2",
			gradient_accumulation_steps=4,
			gradient_checkpointing=True,
			dataloader_num_workers=4,
			dataloader_prefetch_factor=2,
			eval_steps=600,
			evaluation_strategy="steps",
		)

		self.weight_regularization = weight_regularization
		self.batch_normalization = batch_normalization

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

	def train_model(self, model, train_dataset, eval_dataset=None):
		# 啟用 Weight Regularization
		if self.weight_regularization > 0:
			for param in model.parameters():
				param.data.mul_(1 - self.weight_regularization)

		# 啟用 Batch Normalization
		if self.batch_normalization:
			model.classifier = torch.nn.Sequential(
				model.classifier,
				torch.nn.BatchNorm1d(5),  # 將參數設置為輸出的類別數
			)

		# 初始化 Trainer
		trainer = Trainer(
			model=model,  # 使用的模型
			args=self.training_args,  # 訓練參數
			train_dataset=train_dataset,  # 訓練數據集
			eval_dataset=eval_dataset,  # 評估數據集
		)

		# 訓練模型，檢查是否為第一次訓練
		trainer.train()
		return trainer

	def evaluate_model(self, model, test_dataset):
		"""
		Evaluate the model using various metrics
		"""
		trainer = Trainer(
			model=model, args=self.training_args, eval_dataset=test_dataset
		)

		predictions = trainer.predict(test_dataset)
		y_pred = np.argmax(predictions.predictions, axis=1)
		y_true = test_dataset.labels

		# Calculate metrics
		accuracy = accuracy_score(y_true, y_pred)
		precision = precision_score(y_true, y_pred, average="macro")
		recall = recall_score(y_true, y_pred, average="macro")
		f1 = f1_score(y_true, y_pred, average="macro")

		return {
			"accuracy": accuracy,
			"precision": precision,
			"recall": recall,
			"f1": f1,
		}

	@staticmethod
	def save_model(trainer, model_path):
		# 儲存訓練好的模型
		trainer.save_model(model_path, weights_only=True)


# 載入 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained(
	"nlptown/bert-base-multilingual-uncased-sentiment"
)

# Main Program
if __name__ == "__main__":
	# 獲取當前腳本目錄
	script_dir = Path(__file__).resolve().parent.parent.parent.parent

	# 拼接相對路徑
	input_file = "data/custom_data/data_250k.json"  # 原始 JSON 文件

	# 檢查設備並設置批次大小
	device_type = DeviceManager.check_device()
	train_batch_size = DeviceManager.setting_batch_size(device_type)
	print("訓練批次大小:", train_batch_size)

	# 讀取所有數據
	train_texts = []
	train_labels = []
	for batch_texts, batch_labels in DataLoader.read_data_in_batches(
		input_file
	):
		train_texts.extend(batch_texts)
		train_labels.extend(batch_labels)
		print(f"已處理 {len(train_texts)} 條數據")

	# 轉換為numpy數組用於K-fold
	x = np.array(train_texts)
	y = np.array(train_labels)

	# 初始化K-fold
	kf = KFold(n_splits=5, shuffle=True, random_state=42)

	# 初始化評估指標
	sum_accuracy = 0
	sum_precision = 0
	sum_recall = 0
	sum_f1 = 0

	# 初始化訓練器
	sentiment_trainer = SentimentTrainer(
		output_dir=script_dir / "service/model/results",
		logging_dir=script_dir / "service/model/logs",
		train_batch_size=train_batch_size,
		num_train_epochs=3,
		weight_regularization=0.001,
		batch_normalization=True,
	)

	# 執行K-fold交叉驗證
	for fold, (train_index, test_index) in enumerate(kf.split(x)):
		print(f"\nFold {fold + 1}/5")
		X_train, X_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		print(
			f"Train on {len(X_train)} samples, test on {len(X_test)} samples"
		)

		# 創建數據集
		train_dataset = SentimentDataset(
			X_train, y_train, tokenizer, max_length=128
		)
		test_dataset = SentimentDataset(
			X_test, y_test, tokenizer, max_length=128
		)

		# 初始化模型
		model = AutoModelForSequenceClassification.from_pretrained(
			"nlptown/bert-base-multilingual-uncased-sentiment",
			num_labels=5,
			torch_dtype=torch.float32,
		)
		model = model.to("cuda" if torch.cuda.is_available() else "cpu")

		# 訓練模型
		trainer_instance = sentiment_trainer.train_model(
			model, train_dataset, test_dataset
		)

		# 評估模型
		metrics = sentiment_trainer.evaluate_model(model, test_dataset)

		# 累計指標
		sum_accuracy += metrics["accuracy"]
		sum_precision += metrics["precision"]
		sum_recall += metrics["recall"]
		sum_f1 += metrics["f1"]

		# 輸出當前fold結果
		print(f"Fold {fold + 1} Results:")
		print(f'Accuracy: {metrics["accuracy"]:.4f}')
		print(f'Precision: {metrics["precision"]:.4f}')
		print(f'Recall: {metrics["recall"]:.4f}')
		print(f'F1-score: {metrics["f1"]:.4f}')

	# 輸出平均指標
	print("\nAverage Results:")
	print(f"Average Accuracy: {sum_accuracy / 5:.4f}")
	print(f"Average Precision: {sum_precision / 5:.4f}")
	print(f"Average Recall: {sum_recall / 5:.4f}")
	print(f"Average F1-score: {sum_f1 / 5:.4f}")
