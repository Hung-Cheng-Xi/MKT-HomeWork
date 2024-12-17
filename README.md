# Sentiment Analysis Project

這是一個使用 BERT 模型進行情感分析的專案，專案使用了 Hugging Face 的 Transformers 庫和 PyTorch。

## 環境設置

### poetry setting

1. 安裝 Python 3.12.7
2. 使用 Poetry 安裝依賴：

```sh
poetry install
```

### env setting

```sh
python -m venv myenv
source myenv/bin/activate # or env\Scripts\activate

pip install -r requirements.txt
```

## 訓練模型

在 `service/sentiment_analysis.py` 中定義了模型的訓練過程。要訓練模型，請運行以下命令：

```sh
python -m service.model.sentiment_analysis
```

訓練完成後，模型將會儲存在 service/sentiment_model/ 目錄中。

## 預測情感

在 service/predict.py 中定義了情感預測的過程。要進行情感預測，請運行以下命令：

```sh
python -m service.model.predict
```

該腳本將會對一些測試句子進行情感分析並輸出結果。

## 檢查標籤

在 service/check_tag.py 中可以檢查模型的標籤映射。要查看標籤映射，請運行以下命令：

```sh
python -m service.model.check_tag
```

## 依賴項

依賴項定義在 requirements.txt 和 pyproject.toml 中。主要依賴項包括：

- transformers
- torch
- torchvision
- aiofiles
- orjson
- accelerate
- torchaudio
- scikit-learn

## 版本控制

請確保 .gitignore 文件中包含了以下內容，以避免將不必要的文件提交到版本控制系統：

```toml
# yelp data
data/

# training model
results/
sentiment_model/
```
