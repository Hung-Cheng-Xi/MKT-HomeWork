import json, torch, os
from pathlib import Path
from typing import Generator, List, Tuple
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
            return 128  # 其他 GPU 設置中等大小的批次


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


class SentimentTrainer:
    def __init__(
        self,
        output_dir: str,
        logging_dir: str,
        num_train_epochs: int = 1,
        train_batch_size: int = 4,
        save_steps: int = 100,
        save_total_limit: int = 2,
        logging_steps: int = 10,
        load_best_model_at_end: bool = False,
        warmup_steps: int = 10000,
        weight_decay: float = 0.01,
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
            load_best_model_at_end=load_best_model_at_end,  # 訓練結束後載入最佳模型
            warmup_steps=warmup_steps,  # 預熱步數
            weight_decay=weight_decay,  # 權重衰減
            fp16=True,
            fp16_opt_level="O2",
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            dataloader_prefetch_factor=2,
            eval_steps=100,
            evaluation_strategy="steps",
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

    def train_model(self, model, train_dataset, eval_dataset=None):
        # 初始化 Trainer
        trainer = Trainer(
            model=model,  # 使用的模型
            args=self.training_args,  # 訓練參數
            train_dataset=train_dataset,  # 訓練數據集
            eval_dataset=eval_dataset,  # 評估數據集
        )

        # 訓練模型，檢查是否為第一次訓練
        if self._check_first_training():
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train(resume_from_checkpoint=False)

        return trainer

    def evaluate_model(self, model, test_dataset):
        """
        Evaluate the model using various metrics
        """
        trainer = Trainer(
            model=model,
            args=self.training_args,
            eval_dataset=test_dataset
        )
        
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = test_dataset.labels
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
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
    script_dir = Path(__file__).resolve().parent.parent.parent

    # 拼接相對路徑
    input_file = 'data/custom_data/data_250k.json'  # 原始 JSON 文件

    # 檢查設備並設置批次大小
    device_type = DeviceManager.check_device()
    train_batch_size = DeviceManager.setting_batch_size(device_type)
    print("訓練批次大小:", train_batch_size)

    # 讀取所有數據
    train_texts = []
    train_labels = []
    for batch_texts, batch_labels in DataLoader.read_data_in_batches(input_file):
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
    )

    # 執行K-fold交叉驗證
    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        print(f"\nFold {fold + 1}/5")
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(f"Train on {len(X_train)} samples, test on {len(X_test)} samples")

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
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # 訓練模型
        trainer_instance = sentiment_trainer.train_model(
            model, train_dataset, test_dataset
        )

        # 評估模型
        metrics = sentiment_trainer.evaluate_model(model, test_dataset)

        # 累計指標
        sum_accuracy += metrics['accuracy']
        sum_precision += metrics['precision']
        sum_recall += metrics['recall']
        sum_f1 += metrics['f1']

        # 輸出當前fold結果
        print(f'Fold {fold + 1} Results:')
        print(f'Accuracy: {metrics["accuracy"]:.4f}')
        print(f'Precision: {metrics["precision"]:.4f}')
        print(f'Recall: {metrics["recall"]:.4f}')
        print(f'F1-score: {metrics["f1"]:.4f}')

        # 保存模型
        #sentiment_trainer.save_model(
        #    trainer_instance,
        #    script_dir / f"service/model/sentiment_model_fold_{fold + 1}",
        #)

    # 輸出平均指標
    print('\nAverage Results:')
    print(f'Average Accuracy: {sum_accuracy / 5:.4f}')
    print(f'Average Precision: {sum_precision / 5:.4f}')
    print(f'Average Recall: {sum_recall / 5:.4f}')
    print(f'Average F1-score: {sum_f1 / 5:.4f}')