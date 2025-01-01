import json
from pathlib import Path
from typing import List
import nltk
import torch
from transformers import AutoModelForSequenceClassification, BertTokenizer
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from ..insta_food import InstaFoodNER

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
			max_length=512,
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
		return score


def get_test_data():
    reviews = []
    with open('data/custom_data/data_75k.json', 'r') as file:
        content = '[' + file.read().replace('}\n{', '},{') + ']'
        data = json.loads(content)
        reviews = [review['text'] for review in data]
    return reviews

def convert_entities_to_list(text, entities: list[dict]) -> list[str]:
        ents = []
        for ent in entities:
            e = {"start": ent["start"], "end": ent["end"], "label": ent["entity_group"]}
            if ents and -1 <= ent["start"] - ents[-1]["end"] <= 1 and ents[-1]["label"] == e["label"]:
                ents[-1]["end"] = e["end"]
                continue
            ents.append(e)

        return [text[e["start"]:e["end"]] for e in ents]

if __name__ == "__main__":
	# 初始化預測器
	sen_predictor = SentimentPredictor("service/model/k_fold_3")
	food_tokenizer = AutoTokenizer.from_pretrained("Dizex/InstaFoodRoBERTa-NER")
	food_model = AutoModelForTokenClassification.from_pretrained("Dizex/InstaFoodRoBERTa-NER")
	food_predictor = pipeline("ner", model=food_model, tokenizer=food_tokenizer)
	# 獲取測試數據
	test_sentences = get_test_data()
	results = []
	p_count = 1
	# Run Program
	# 讀取完整評論
	for sentences in test_sentences:
		# 評論分段，依段落
		if p_count % 10 == 0:
			print("Now Process: " + p_count)
		sentences_part = nltk.sent_tokenize(sentences)
		for part in sentences_part:
			food_result = food_predictor(part, aggregation_strategy="simple")
			# 如果有讀到食物相關資訊
			if len(food_result) > 0:
				food_type = convert_entities_to_list(part, food_result)
				food_rank = sen_predictor.predict(part) + 1
				for food in food_type:
					results.append({
						'food': food,
						'rank': food_rank,
					})
		if p_count % 1000 == 0:
			with open('data/custom_data/output.json', 'w') as f:
				json.dump(results, f)
		p_count += 1
	
	
    
    