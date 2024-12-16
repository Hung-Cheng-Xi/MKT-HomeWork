import json
from pathlib import Path
from typing import List

import torch
from compare_models import compare_models, plot_comparison, plot_distribution
from transformers import AutoModelForSequenceClassification, BertTokenizer


class SentimentPredictor:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self.model.eval()  # 設置為評估模式

    def predict(self, text):
        # 對輸入文本進行編碼
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # 獲取模型預測以及禁止梯度計算
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )

        # 獲取預測結果
        predictions = torch.softmax(outputs.logits, dim=1)
        score = torch.argmax(predictions, dim=1).item()

        # 將分數轉換為情感描述
        sentiment_map = {
            0: "非常負面",
            1: "負面",
            2: "中性",
            3: "正面",
            4: "非常正面",
        }

        return {
            "score": score,
            "sentiment": sentiment_map[score],
            "probabilities": predictions[0].tolist(),
        }


def get_test_data() -> List:
    # 獲取當前腳本目錄
    script_dir = Path(__file__).resolve().parent

    # 拼接相對路徑
    input_file = (
        script_dir.parent / "data" / "custom_data" / "train.json"
    )  # 原始 JSON 文件

    # 讀取 JSON 文件並提取訓練數據
    test_sentences = []
    with open(input_file, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):

            if i > 5:
                break

            review = json.loads(line)
            test_sentences.append(review["text"])
            if (i + 1) % 100000 == 0:
                print(f"已處理 {i + 1} 條數據")
    print("數據讀取完成")

    return test_sentences


if __name__ == "__main__":
    # 初始化兩個預測器
    predictor1 = SentimentPredictor(
        "nlptown/bert-base-multilingual-uncased-sentiment"
    )
    predictor2 = SentimentPredictor("./sentiment_model")

    # 獲取測試數據
    test_sentences = (
        get_test_data()
    )  # 確保 test_sentences 是從 get_test_data 函數中返回的列表

    # 比較兩個模型
    model1_results, model2_results = compare_models(
        predictor1, predictor2, test_sentences
    )

    # 繪製散點圖比較
    plot_comparison(
        test_sentences,
        model1_results,
        model2_results,
        save_path="comparison_scatter_plot.png",
    )

    # 繪製分佈圖比較
    plot_distribution(
        model1_results,
        model2_results,
        save_path="comparison_distribution_plot.png",
    )
