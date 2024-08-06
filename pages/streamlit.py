import re
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    数据预处理函数，从输入的数据中提取电阻和角度信息
    :param data: 输入的数据字符串
    :return: 处理后的 NumPy 数组，包含电阻和角度信息，若数据无效则返回 None
    """
    # 按行读取数据
    lines = data.splitlines()

    # 处理数据集，提取电阻和角度
    processed_data = []
    for line in lines:
        match = re.search(r'电阻:(\d+\.\d+) 角度:(\d+\.\d+)', line)
        if match:
            resistance = float(match.group(1))
            angle = float(match.group(2))
            processed_data.append([resistance, angle])

    # 转换为 NumPy 数组
    if processed_data:
        return np.array(processed_data)
    else:
        return None

def check_if_dynamic(processed_data):
    """
    判断数据是否表示动态手势，根据电阻值大于阈值来判断
    :param processed_data: 处理后的数据数组
    :return: 布尔值，表示是否为动态手势
    """
    # 假设动态手势的电阻值大于某个阈值
    threshold = 30000
    return np.any(processed_data[:, 0] > threshold)

def action_ended():
    """
    判断动作是否结束，根据数据列表中的数据数量达到阈值来判断
    :return: 布尔值，表示动作是否结束
    """
    # 假设当数据列表中的数据数量达到一定值时，动作结束
    threshold = 10
    return len(data_list) >= threshold

def scale_data(data_list):
    """
    使用 StandardScaler 对数据进行缩放
    :param data_list: 数据列表
    :return: 缩放后的数据
    """
    # 使用 StandardScaler 进行数据缩放
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(data_list))
    return scaled_data
  

