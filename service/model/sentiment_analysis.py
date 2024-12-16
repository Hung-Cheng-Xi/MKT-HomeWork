import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

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


def setting_batch_size() -> int:
    # 檢查是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 動態設置 batch size
    default_batch_size = 4
    max_batch_size = (
        64 if device.type == "cuda" else 8
    )  # GPU 最大批次大小較高，CPU 较低
    return min(default_batch_size, max_batch_size)


def get_train_data() -> Tuple[List, List]:
    # 獲取當前腳本目錄
    script_dir = Path(__file__).resolve().parent

    # 拼接相對路徑
    input_file = (
        script_dir.parent.parent / "data" / "custom_data" / "train.json"
    )  # 原始 JSON 文件

    # 讀取 JSON 文件並提取訓練數據
    train_texts = []
    train_labels = []
    with open(input_file, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):

            if i > 1000:
                break

            review = json.loads(line)
            train_texts.append(review["text"])
            train_labels.append(int(review["stars"]))
            if (i + 1) % 100000 == 0:
                print(f"已處理 {i + 1} 條數據")
    print("數據讀取完成")

    return train_texts, train_labels


# 載入 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(
        "nlptown/bert-base-multilingual-uncased-sentiment", num_labels=5
    )

    # 獲取訓練數據
    train_texts, train_labels = get_train_data()
    train_dataset = SentimentDataset(
        train_texts, train_labels, tokenizer, max_length=128
    )

    train_batch_size = setting_batch_size()

    # 設置訓練參數
    training_args = TrainingArguments(
        output_dir="./results",  # 訓練結果儲存路徑
        num_train_epochs=1,  # 訓練輪數
        per_device_train_batch_size=train_batch_size,  # 每次訓練的批次大小
        per_device_eval_batch_size=train_batch_size,  # 驗證時的批次大小
        warmup_steps=500,  # 預熱步數
        weight_decay=0.01,  # 正則化
        logging_dir="./logs",  # 訓練過程日誌儲存路徑
        logging_steps=10,  # 每多少步記錄一次日誌
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,  # 使用的模型
        args=training_args,  # 訓練參數
        train_dataset=train_dataset,  # 訓練數據集
    )

    # 訓練模型
    trainer.train()

    # 儲存訓練好的模型
    trainer.save_model("./sentiment_model")

    # 評估模型
    # 在此範例中，假設有測試數據集
    test_texts = ["這是很棒的經驗", "非常糟糕的服務"]
    test_labels = [4, 0]
    test_dataset = SentimentDataset(
        test_texts, test_labels, tokenizer, max_length=128
    )

    # 設定評估參數並進行評估
    trainer.evaluate(test_dataset)
