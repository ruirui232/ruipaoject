import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from openpyxl.workbook import Workbook

a1 = np.array([4.6245,
               11.9166,
               5.5571,
               6.7524,
               3.8386,
               6.1218,
               2.9699,
               9.6271,
               3.0139,
               4.4735,
               4.8722,
               7.2063
               ])
A = []
# A.append(a1)
for i, ia in enumerate(a1):
    sa = a1 / ia
    sa[i] = 0
    A.append(sa)
A = np.array(A).T
a = pd.DataFrame(A)
# a.to_csv(r'C:\Users\Dell\OneDrive\Python\Project  ts/test.csv')

A1 = np.mat(a1)
A1 = np.transpose(A1)
A2 = A1 / A1[0]
A3 = A1 / A1.item(1)
A4 = A1 / A1.item(2)
A5 = A1 / A1.item(3)
A6 = A1 / A1.item(4)
A7 = A1 / A1.item(5)
A8 = A1 / A1.item(6)
A9 = A1 / A1.item(7)
A10 = A1 / A1.item(8)
A11 = A1 / A1.item(9)
A12 = A1 / A1.item(10)
A13 = A1 / A1.item(11)
# A=np.append(A2,A3,A4,axis=1)
A_a = A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10 + A11 + A12 + A13
A_1 = A_a - 1
A_max = np.max(A_1)

B = A / A_max
b = pd.DataFrame(B)
# b.to_csv(r'C:\Users\Dell\OneDrive\Python\Project  ts/test1.csv')
data: object = pd.read_excel('E:\project\ju ZHEN\I.xls')
I = data.values  # 单位矩阵
C = I - B
C1 = np.linalg.inv(C)  # 算出了逆矩阵
T = B * C1
R = np.sum(T, axis=0)  # 各列之和
D = np.sum(T, axis=1)  # 各行之和
X = D + R
Y = D - R
Y_min = np.min(Y)
Y_max = np.max(Y)
X_ave = np.average(X)
X_std = np.std(X)
X_sum = X_ave + X_std

T = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "b1", "b2", "b3", "c1", "c2"]
Xm = np.mean(X)  # 到了添加0刻线和阈值这一步

fig = plt.figure(figsize=(8, 6))  # 新建画布
ax = plt.subplot(1, 1, 1)  # 子图初始化
ax.scatter(X, Y)  # 绘制散点图
plt.vlines(X_sum, Y_min, Y_max)  # 绘制一条竖直的线
plt.title('verify  sheet 33', size=20, loc='right')  # 给图添加标题
for i in range(len(Y)):
    ax.text(X[i], Y[i] + 0.01, T[i], fontsize=12, color="r", style="italic", weight="light",
            verticalalignment='center', horizontalalignment='right', rotation=0)  # 给散点加标签
plt.savefig(r'C:\Users\Dell\OneDrive\Python\Project  ts\ju ZHEN/verify sheet 33.png')
plt.show()