import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.workbook import Workbook

a1 = np.array([6.9067,
11.3537,
5.3456,
3.516,
3.8054,
6.8089,
6.8397,
8.8213,
5.0062,
3.9234,
4.1707,
7.2309
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
print(X_sum)
T = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "b1", "b2", "b3", "c1", "c2"]
Xm = np.mean(X)  # 到了添加0刻线和阈值这一步

fig = plt.figure(figsize=(8, 6))  # 新建画布
ax = plt.subplot(1, 1, 1)  # 子图初始化
ax.scatter(X, Y)  # 绘制散点图
plt.vlines(X_sum, Y_min, Y_max)  # 绘制一条竖直的线
plt.title('verify  sheet 3', size=20, loc='right')  # 给图添加标题
for i in range(len(Y)):
    ax.text(X[i], Y[i] + 0.01, T[i], fontsize=12, color="r", style="italic", weight="light",
            verticalalignment='center', horizontalalignment='right', rotation=0)  # 给散点加标签
plt.savefig(r'C:\Users\Dell\OneDrive\Python\Project  ts\ju ZHEN/verify sheet 3.png')
plt.show()
# A23=np.append(A2,A3,axis=1)
# A45=np.append(A4,A5,axis=1)
# A67=np.append(A6,A7,axis=1)
# A89=np.append(A8,A9,axis=1)
# A1011=np.append(A10,A11,axis=1)
# A1213=np.append(A12,A13,axis=1)
# A2345=np.append(A23,A45,axis=1)
# A6789=np.append(A67,A89,axis=1)
# A10_13=np.append(A1011,A1213,axis=1)
# A2_9=np.append(A2345,A6789,axis=1)
# A_Z=np.append(A2_9,A10_13,axis=1)
# A_0=np.where(A_Z=1,A_Z,0)

# print(A_0)


# A_Z=np.mat(A_Z)
# data = pd.DataFrame(A_Z)
# writer = pd.ExcelWriter('A_Z.xlsx')		# 写入Excel文件
# data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer.save()
# writer.close()
