import numpy as np
import pandas as pd
import re
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import dump

# 假设你有多个数据集，每个数据集对应一种手势
data_sets = {
    1: 'F:\PyCharm\Py_Projects\静态手指数字1表示数据集.txt',
    9: 'F:\PyCharm\Py_Projects\数字9数据集.txt',
    # 添加更多手势的数据集路径
}

all_data = []
all_labels = []

for label, data_path in data_sets.items():
    # 按行读取数据
    with open(data_path, 'r') as f:
        lines = f.readlines()

    # 处理数据集，提取电阻和角度
    processed_data = []
    for line in lines:
        match = re.search(r'电阻:(\d+\.\d+) 角度:(\d+\.\d+)', line)
        if match:
            resistance = float(match.group(1))
            angle = float(match.group(2))
            processed_data.append([resistance, angle])

    # 转换为 DataFrame
    if processed_data:
        df = pd.DataFrame(processed_data, columns=['Resistance', 'Angle'])
        all_data.append(df[['Resistance', 'Angle']].values)
        all_labels.extend([label] * len(df))
    else:
        print(f"数据集中 {data_path} 没有有效数据")

# 合并所有数据
if all_data:
    X = np.concatenate(all_data, axis=0)
else:
    X = None
y = np.array(all_labels)

if X is not None:
    # 打印 X 和 y 的形状
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练 SVM 模型
    svm_model = svm.SVC(kernel='linear')
    svm_model.fit(X_train, y_train)

    # 保存 SVM 模型
    dump(svm_model, 'F:\PyCharm\Py_Projects\svm_model.joblib')  # 指定保存路径

    # 在测试集上进行预测
    svm_pred = svm_model.predict(X_test)

    # 评估模型性能
    svm_accuracy = accuracy_score(y_test, svm_pred)

    print("SVM Accuracy:", svm_accuracy)
else:
    print("数据集中没有有效数据，无法进行 SVM 模型训练和评估")