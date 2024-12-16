from transformers import AutoModelForSequenceClassification, BertTokenizer

# 加載模型和標籤映射
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=5
    )

    # 查看標籤映射
    print(model.config.id2label)
