# /chapter/4_7_1_MLP.ipynb
# /chapter/4_7_1_MLP.ipynb
# from keras.preprocessing import sequence
# from keras.utils import multi_gpu_model
# from keras import regularizers  # 正则化
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from keras.datasets import boston_housing

(x_train, y_train), (x_valid, y_valid) = boston_housing.load_data()  # 加载数据

# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)

# print(x_train_pd.head(5))
# print('-------------------')
# print(y_train_pd.head(5))

# # 训练集归一化
# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit(x_train_pd)
# x_train = min_max_scaler.transform(x_train_pd)
# print(x_train)
#
# min_max_scaler.fit(y_train_pd)
# y_train = min_max_scaler.transform(y_train_pd)
#
# # 验证集归一化
# min_max_scaler.fit(x_valid_pd)
# x_valid = min_max_scaler.transform(x_valid_pd)
#
# min_max_scaler.fit(y_valid_pd)
# y_valid = min_max_scaler.transform(y_valid_pd)
#
# # 单CPU or GPU版本，若有GPU则自动切换
# model = Sequential()  # 初始化，很重要！
# model.add(Dense(units=10,  # 输出大小
#                 activation='relu',  # 激励函数
#                 input_shape=(x_train_pd.shape[1],)  # 输入大小, 也就是列的大小
#                 )
#           )
#
# model.add(Dropout(0.2))  # 丢弃神经元链接概率
#
# model.add(Dense(units=15,
#                 #                 kernel_regularizer=regularizers.l2(0.01),  # 施加在权重上的正则项
#                 #                 activity_regularizer=regularizers.l1(0.01),  # 施加在输出上的正则项
#                 activation='relu'  # 激励函数
#                 # bias_regularizer=keras.regularizers.l1_l2(0.01)  # 施加在偏置向量的正则项
#                 )
#           )
#
# model.add(Dense(units=1,
#                 activation='linear'  # 线性激励函数 回归一般在输出层用这个激励数
#                 )
#           )
#
# print(model.summary())  # 打印网络层次结构
#
# model.compile(loss='mse',  # 损失均方误差
#               optimizer='adam',  # 优化器
#               )
# history = model.fit(x_train, y_train,
#                     epochs=200,  # 迭代次数
#                     batch_size=200,  # 每次用来梯度下降的批处理数据大小
#                     verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，输出训练进度，2：输出每一个epoch
#                     validation_data=(x_valid, y_valid)  # 验证集
#                     )
#
# import matplotlib.pyplot as plt
#
# # 绘制训练 & 验证的损失值
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# from keras.utils import plot_model
# from keras.models import load_model
#
# # 保存模型
# model.save('model_MLP.h5')  # 生成模型文件 'my_model.h5'
#
# # 模型可视化 需要安装pydot pip install pydot
# plot_model(model, to_file='model_MLP.png', show_shapes=True)
#
# # 加载模型
# model = load_model('model_MLP.h5')
















