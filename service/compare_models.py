import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 使用 Seaborn 主題
sns.set_theme(style="whitegrid")

def compare_models(predictor1, predictor2, sentences):
    model1_results = []
    model2_results = []

    # 獲取兩個模型的預測結果
    for sentence in sentences:
        result1 = predictor1.predict(sentence)
        result2 = predictor2.predict(sentence)

        model1_results.append(result1)
        model2_results.append(result2)

    return model1_results, model2_results

def plot_comparison(sentences, model1_results, model2_results, save_path=None):
    model1_scores = [res["score"] + 1 for res in model1_results]  # 模型 1 的情感分數
    model2_scores = [res["score"] + 1 for res in model2_results]  # 模型 2 的情感分數

    # 散點圖
    plt.figure(figsize=(14, 7))
    plt.scatter(range(len(sentences)), model1_scores, color="blue", label="Model 1", alpha=0.7)
    plt.scatter(range(len(sentences)), model2_scores, color="red", label="Model 2", alpha=0.7)

    # 標題與標籤
    plt.xlabel("Text Samples", fontsize=14)
    plt.ylabel("Sentiment Score", fontsize=14)
    plt.title("Sentiment Score Comparison (Scatter Plot)", fontsize=16)
    plt.xticks(range(len(sentences)), [f"Text {i+1}" for i in range(len(sentences))], rotation=45)
    plt.legend(fontsize=12)

    # 增加網格
    plt.grid(axis="both", linestyle="--", alpha=0.7)

    # 添加緊湊布局
    plt.tight_layout()

    # 保存圖表（如果指定保存路徑）
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"圖表已保存至: {save_path}")

    # 顯示圖表
    plt.show()

def plot_distribution(model1_results, model2_results, save_path=None):
    model1_scores = [res["score"] + 1 for res in model1_results]
    model2_scores = [res["score"] + 1 for res in model2_results]

    plt.figure(figsize=(12, 6))

    sns.kdeplot(model1_scores, shade=True, color="blue", label="Model 1", alpha=0.7)
    sns.kdeplot(model2_scores, shade=True, color="red", label="Model 2", alpha=0.7)

    # 標題與標籤
    plt.xlabel("Sentiment Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("Sentiment Score Distribution", fontsize=16)
    plt.legend(fontsize=12)

    # 添加緊湊布局
    plt.tight_layout()

    # 保存圖表（如果指定保存路徑）
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"圖表已保存至: {save_path}")

    # 顯示圖表
    plt.show()
