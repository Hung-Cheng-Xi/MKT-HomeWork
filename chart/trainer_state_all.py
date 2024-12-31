import json
import os
import matplotlib.pyplot as plt

# 使用相對路徑，從目前的程式目錄尋找 JSON 檔案
file_name = "trainer_state.json"  # JSON 檔案名稱
file_path = os.path.join(os.path.dirname(__file__), file_name)

# 讀取 JSON 檔案
try:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 輸出讀取的內容
    print("JSON 檔案內容:")
    #print(json.dumps(data, indent=4, ensure_ascii=False))
except FileNotFoundError:
    print(f"檔案未找到: {file_path}")
except json.JSONDecodeError as e:
    print(f"JSON 格式錯誤: {e}")

# 提取 log_history 中的 loss 和 step
log_history = data.get("log_history", [])
steps = [entry["step"] for entry in log_history if "step" in entry and "loss" in entry]
losses = [entry["loss"] for entry in log_history if "step" in entry and "loss" in entry]

# 繪製圖表
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, label="Loss")

# 添加標題和標籤
plt.title("Training Loss vs Steps", fontsize=16)
plt.xlabel("Steps", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)

current_dir = os.path.dirname(os.path.abspath(__file__))  # 取得執行檔案所在的目錄
output_path = os.path.join(current_dir, "trainer_state_all.png")
plt.savefig(output_path)  # 儲存圖片到指定路徑

# 顯示圖表
plt.show()