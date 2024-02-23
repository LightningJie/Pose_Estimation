# 导入必要的库  
from keras.models import Sequential  # 导入Sequential模型，用于构建层叠模型  
from keras.layers import LSTM, Dense  # 导入LSTM和Dense层，用于构建神经网络  
from keras.callbacks import TensorBoard  # 导入TensorBoard回调，用于可视化训练过程  
import numpy as np  # 导入numpy库，用于数组操作  
import os  # 导入os库，用于文件路径操作  
from sklearn.model_selection import train_test_split  # 导入train_test_split函数，用于划分数据集  
from keras.utils import to_categorical  # 导入to_categorical函数，用于将标签转换为one-hot编码  

# 设置TensorBoard日志目录  
log_dir = os.path.join('Logs')  # 使用os.path.join创建日志目录的路径  
tb_callback = TensorBoard(log_dir=log_dir)  # 初始化TensorBoard回调  

# 定义模型参数  
no_sequences = 2  # 每个动作的序列数
sequence_length = 30  # 每个序列的帧数  
DATA_PATH = os.path.join('MP_Data')  # 数据集的路径  
actions = np.array(['666', '888', ])  # 可能的动作类别列表

# 创建标签映射  
label_map = {label: num for num, label in enumerate(actions)}  # 使用字典推导式创建标签到整数的映射

# 加载和预处理数据  
sequences, labels = [], []  # 初始化序列和标签列表  
for action in actions:  # 遍历每个动作类别  
    for sequence in range(no_sequences):  # 遍历每个动作类别的序列数  
        window = []  # 初始化当前序列的窗口（即帧的列表）  
        for frame_num in range(sequence_length):  # 遍历每个序列的帧数  
            # 加载帧数据并添加到窗口中  
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)  # 将窗口添加到序列列表中  
        labels.append(label_map[action])  # 将当前动作类别的标签添加到标签列表中  

# 数据预处理  
X = np.array(sequences)  # 将序列列表转换为numpy数组  
y = to_categorical(labels).astype(int)  # 将标签列表转换为one-hot编码并转换为整数类型  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)  # 划分训练集和测试集  

# 构建LSTM模型  
model = Sequential()  # 初始化序贯模型
#关于input_shape，原作者的网络是(30,1662),1662=33*4 + 468*2 + 21*3 + 21*3，而我只需要人体姿态坐标，故只有132
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 132)))  # 添加第一个LSTM层，设置单元数、是否返回序列、激活函数和输入形状
model.add(LSTM(128, return_sequences=True, activation='relu'))  # 添加第二个LSTM层  
model.add(LSTM(64, return_sequences=False, activation='relu'))  # 添加第三个LSTM层，设置不返回序列  
model.add(Dense(64, activation='relu'))  # 添加第一个全连接层  
model.add(Dense(32, activation='relu'))  # 添加第二个全连接层  
model.add(Dense(actions.shape[0], activation='softmax'))  # 添加输出层，设置单元数、激活函数和动作类别的数量  

# 编译模型  
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])  # 设置优化器、损失函数和评估指标  

# 训练模型  
model.fit(X_train, y_train, epochs=300, callbacks=[tb_callback])  # 训练模型，设置训练轮数和使用TensorBoard回调

# 显示模型总结  
model.summary()  # 打印模型的架构和参数信息  

# 保存模型  
model.save('action.h5')  # 将训练好的模型保存为HDF5文件