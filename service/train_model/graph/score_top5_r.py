import matplotlib.pyplot as plt
import json
from collections import defaultdict

# 讀取資料
with open('data/custom_data/output.json', 'r', encoding='utf-8') as f:
   data = json.load(f)

# 計算每種食物的平均評分
food_counts = defaultdict(int)
food_ratings = defaultdict(list)
for item in data:
   food_counts[item['food']] += 1
   food_ratings[item['food']].append(item['rank'])

# 只計算超過100筆的食物
avg_ratings = {food: sum(ratings)/len(ratings) 
             for food, ratings in food_ratings.items() 
             if food_counts[food] >= 10}
# 取平均分數前5高的食物
top_5 = sorted(avg_ratings.items(), key=lambda x: x[1])[:5]

# 準備繪圖資料
foods = [item[0] for item in top_5]
scores = [item[1] for item in top_5]


# 繪圖
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
plt.ylim(0, max(scores) + 1)  # 設定合適的y軸範圍
plt.bar(foods, scores)

plt.title(f'評分倒數的食物', fontsize=12)
plt.xlabel('食物', fontsize=10)
plt.ylabel('平均評分', fontsize=10)
plt.xticks(rotation=45)
plt.legend()
plt.savefig('data/custom_data/score_top5_r.png')
plt.tight_layout()
