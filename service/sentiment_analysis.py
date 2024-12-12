from transformers import pipeline
import json

# 加載指定的情感分析模型
model_name = "juliensimon/reviews-sentiment-analysis"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

# 自定義標籤轉換
label_mapping = {0: 'NEGATIVE', 1: 'POSITIVE'}

# 單筆 review.json 資料
review_data = {
    "review_id": "KU_O5udG6zpxOg-VcAEodg",
    "user_id": "mh_-eMZ6K5RLWhZyISBhwA",
    "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
    "stars": 3.0,
    "useful": 0,
    "funny": 0,
    "cool": 0,
    "text": "If you decide to eat here, just be aware it is going to take about 2 hours from beginning to end. We have tried it multiple times, because I want to like it! I have been to it's other locations in NJ and never had a bad experience. \n\nThe food is good, but it takes a very long time to come out. The waitstaff is very young, but usually pleasant. We have just had too many experiences where we spent way too long waiting. We usually opt for another diner or restaurant on the weekends, in order to be done quicker.",
    "date": "2018-07-07 22:09:11"
}

# 提取評論文本
review_text = review_data['text']

# 對評論文本進行情感分析
result = sentiment_pipeline(review_text)

# 轉換標籤
for item in result:
    item['label'] = label_mapping[int(item['label'].split('_')[-1])]

# 添加情感分析結果到 review_data
review_data['sentiment_label'] = result[0]['label']
review_data['sentiment_score'] = result[0]['score']

# 輸出結果
print(json.dumps(review_data, indent=4))
