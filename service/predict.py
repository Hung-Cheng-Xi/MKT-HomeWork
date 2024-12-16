import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer

from compare_models import compare_models, plot_comparison  # 匯入比較和繪圖功能
from compare_models import compare_models, plot_comparison, plot_distribution
from predict import SentimentPredictor, get_test_data


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
    test_sentences = ["hello world!!", "fuck you!!"]
    return test_sentences


if __name__ == "__main__":
    # 初始化兩個預測器
    predictor1 = SentimentPredictor("nlptown/bert-base-multilingual-uncased-sentiment")
    predictor2 = SentimentPredictor("./sentiment_model")

    # 獲取測試數據
    test_sentences = get_test_data()  # 確保 test_sentences 是從 get_test_data 函數中返回的列表

    # 比較兩個模型
    model1_results, model2_results = compare_models(predictor1, predictor2, test_sentences)

    # 繪製散點圖比較
    plot_comparison(test_sentences, model1_results, model2_results, save_path="comparison_scatter_plot.png")

    # 繪製分佈圖比較
    plot_distribution(model1_results, model2_results, save_path="comparison_distribution_plot.png")

