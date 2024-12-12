from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加載模型和標籤映射
model_name = "juliensimon/reviews-sentiment-analysis"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 查看標籤映射
print(model.config.id2label)
