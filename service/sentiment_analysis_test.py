from transformers import pipeline

# 加載指定的情感分析模型
model_name = "juliensimon/reviews-sentiment-analysis"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

# 自定義標籤轉換
label_mapping = {0: 'NEGATIVE', 1: 'POSITIVE'}

# 測試單條評論
review = "The food was amazing and the service was excellent!"
result = sentiment_pipeline(review)

# 轉換標籤
for item in result:
    item['label'] = label_mapping[int(item['label'].split('_')[-1])]

print(result)
# 輸出範例: [{'label': 'POSITIVE', 'score': 0.9998762}]
