import numpy as np
import pandas as pd
import re
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import load

# 加载训练好的 SVM 模型和 ANN 模型
svm_model = load('F:\PyCharm\Py_Projects\svm_model.joblib')  # 替换为实际的 SVM 模型路径
ann_model = load('F:\PyCharm\Py_Projects\Ann_model.joblib')  # 替换为实际的 ANN 模型路径

# 假设你有新的数据集
new_data_path = 'F:\PyCharm\Py_Projects\牛.txt'

# 读取新数据集
new_data = []
with open(new_data_path, 'r') as f:
    lines = f.readlines()

# 处理新数据集，提取电阻和角度
processed_new_data = []
for line in lines:
    match = re.search(r'电阻:(\d+\.\d+) 角度:(\d+\.\d+)', line)
    if match:
        resistance = float(match.group(1))
        angle = float(match.group(2))
        processed_new_data.append([resistance, angle])

if processed_new_data:
    X_new = np.array(processed_new_data)

    # 数据标准化
    scaler = StandardScaler()
    scaler.fit(X_new)  # 先进行拟合
    X_new = scaler.transform(X_new)

    # 使用 SVM 模型进行预测
    svm_pred = svm_model.predict(X_new)
    print("SVM 预测结果:", svm_pred)

    # 使用 ANN 模型进行预测
    Ann_pred = ann_model.predict(X_new)
    print("ANN 预测结果:", Ann_pred)
else:
    print("新数据集中没有有效数据")