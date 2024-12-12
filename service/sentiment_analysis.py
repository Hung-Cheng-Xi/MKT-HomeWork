import torch
from transformers import BertTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
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
            padding='max_length',  # Pad to max length
            truncation=True,
            return_attention_mask=True,  # Return attention mask
            return_tensors='pt',  # Return PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 載入 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
# model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment', num_labels=5)

if __name__ == '__main__':
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", num_labels=5)

    # 定義訓練數據
    train_texts = ["我愛這個產品", "這個服務太差了", "品質不錯，會再來", "糟透了，完全不滿意"]
    train_labels = [4, 0, 3, 1]  # 標籤範圍是 0 到 4，表示情感分數
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=128)

    # 設置訓練參數
    training_args = TrainingArguments(
        output_dir='./results',            # 訓練結果儲存路徑
        num_train_epochs=3,                # 訓練輪數
        per_device_train_batch_size=8,     # 每次訓練的批次大小
        per_device_eval_batch_size=16,     # 驗證時的批次大小
        warmup_steps=500,                  # 預熱步數
        weight_decay=0.01,                 # 正則化
        logging_dir='./logs',              # 訓練過程日誌儲存路徑
        logging_steps=10,                  # 每多少步記錄一次日誌
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,                       # 使用的模型
        args=training_args,                # 訓練參數
        train_dataset=train_dataset,       # 訓練數據集
    )

    # 訓練模型
    trainer.train()

    # 儲存訓練好的模型
    trainer.save_model("./sentiment_model")

    # 評估模型
    # 在此範例中，假設有測試數據集
    test_texts = ["這是很棒的經驗", "非常糟糕的服務"]
    test_labels = [4, 0]
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length=128)

    # 設定評估參數並進行評估
    trainer.evaluate(test_dataset)
