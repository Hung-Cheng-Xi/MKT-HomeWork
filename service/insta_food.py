from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class InstaFoodNER:
    _tokenizer = AutoTokenizer.from_pretrained("Dizex/InstaFoodRoBERTa-NER")
    _model = AutoModelForTokenClassification.from_pretrained("Dizex/InstaFoodRoBERTa-NER")
    _pipeline = pipeline("ner", model=_model, tokenizer=_tokenizer)

    @classmethod
    def analyze(cls, text, aggregation_strategy="simple"):
        """
        Perform Named Entity Recognition (NER) on the provided text.

        Args:
            text (str): The text to analyze.
            aggregation_strategy (str): Strategy for aggregating NER results. Default is "simple".

        Returns:
            list: The NER results.
        """
        return cls._pipeline(text, aggregation_strategy=aggregation_strategy)

# # 使用方式
# example = "Today's meal: Fresh olive poké bowl topped with chia seeds. Very delicious!"
# result = InstaFoodNER.analyze(example)
# print(result)
