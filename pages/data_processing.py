import socket
import streamlit as st
from joblib import load
from data_processing import preprocess_data, check_if_dynamic, action_ended, scale_data


# 加载训练好的 SVM 模型和 ANN 模型
svm_model = load('F:\PyCharm\Py_Projects\svm_model.joblib')  # 替换为实际的 SVM 模型路径
ann_model = load('F:\PyCharm\Py_Projects\Ann_model.joblib')  # 替换为实际的 ANN 模型路径

# 初始化套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 8088)  # 这里修改为实际的服务器 IP 地址和端口号
sock.bind(server_address)

data_list = []  # 初始化数据列表

# Streamlit 页面设置
st.title("智能手套数据展示与预测")  # 确保在导入 streamlit 后使用

while True:
    # 接收数据报
    data, address = sock.recvfrom(4096)

    # 数据预处理
    processed_data = preprocess_data(data.decode('utf-8'))  # 假设数据是以 UTF-8 编码发送的

    if processed_data is not None:
        # 判断数据是否来自动态手势
        is_dynamic = check_if_dynamic(processed_data)

        if not is_dynamic:
            # 将数据放入线性分类器（SVM 模型）进行处理
            svm_prediction = svm_model.predict(processed_data)
            # 显示结果
            st.write("SVM 预测结果:", svm_prediction)
        else:
            # 将数据放入列表
            data_list.extend(processed_data)
            # 循环“采集 - 堆栈”该操作，直至动作结束
            if action_ended():
                # 将数据缩放
                scaled_data = scale_data(data_list)
                # 放入 ANN 模型，得出结果
                ann_prediction = ann_model.predict(scaled_data)
                # 显示结果
                st.write("ANN 预测结果:", ann_prediction)
                # 清空数据列表，准备下一次操作
                data_list.clear()

# 绘制折线图（根据实际需求修改）
if len(data_list) > 0:
    resistance_values = [data[0] for data in data_list]
    angle_values = [data[1] for data in data_list]

    fig, ax = plt.subplots()
    ax.plot(resistance_values, label='电阻')
    ax.plot(angle_values, label='角度')
    ax.set_xlabel('数据点')
    ax.set_ylabel('值')
    ax.set_title('电阻和角度随数据点的变化')
    ax.legend()

    # 在 Streamlit 中显示折线图
    st.pyplot(fig)