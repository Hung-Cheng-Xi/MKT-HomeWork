import json
from typing import Generator, List, Tuple

import torch
from torch.utils.data import Dataset


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


class DeviceManager:
	@staticmethod
	def check_device() -> str:
		"""
		判斷當前設備是本地 CPU 還是 Colab T4 GPU。
		Check if the environment is local CPU or Colab T4 GPU.
		:return: Device type ("Local CPU" or "Colab T4 GPU")
		"""
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if device.type == "cpu":
			return "Local CPU"

		gpu_name = torch.cuda.get_device_name(0)
		if "T4" in gpu_name:
			return "Colab T4 GPU"

		return f"Other GPU ({gpu_name})"

	@staticmethod
	def setting_batch_size(device: str) -> int:
		"""
		根據設備類型動態設置批次大小。
		:param device: 設備類型 ("Local CPU" 或 "Colab T4 GPU")
		:return: 訓練批次大小
		"""
		if device == "Local CPU":
			return 4  # 本地 CPU 設置較小的批次大小
		elif device == "Colab T4 GPU":
			return 32  # Colab T4 GPU 可以使用較大的批次大小
		else:
			return 16  # 其他 GPU 設置中等大小的批次


class DataLoader:
	@staticmethod
	def read_data_in_batches(
		input_file: str, batch_size: int = 1000
	) -> Generator[Tuple[List, List], None, None]:
		"""
		分批次讀取 JSON 文件，並將每一批次的文本和標籤返回
		:param input_file: JSON 文件路徑
		:param batch_size: 每次讀取的批次大小
		:return: texts 和 labels 的列表
		"""
		texts = []
		labels = []

		with open(input_file, "r", encoding="utf-8") as file:
			for line in file:
				if len(texts) == batch_size:
					yield texts, labels
					texts, labels = [], []

				review = json.loads(line)
				texts.append(review["text"])
				labels.append(int(review["stars"] - 1))

			if texts:
				yield texts, labels
