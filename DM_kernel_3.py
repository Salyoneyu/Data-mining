import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

# 首先选择鸢尾花的五个数据集
x=np.array([[5.9,3],[6.9,3.1],[6.6,2.9],[4.6,3.2],[6,2.2]])
# 求其线性核矩阵
kernel=linear_kernel(x)
# 生成对角线元素为1的矩阵
a=np.eye(5,k=0)
# 取核矩阵的对角线元素
w=kernel.diagonal()*a
# 对w进行求逆后乘0.5次方
w_n=(np.linalg.inv(w))**(0.5)
# 求矩阵I，在这里表示为A
A=a-(np.ones_like(kernel))*(1/5)
# 求居中核矩阵
z=np.dot(A,kernel)
z2=np.dot(z,A)
print("居中化核矩阵：",z2)
# 求规范核矩阵
kn1=np.dot(w_n,kernel)
kn=np.dot(kn1,w_n)
print("规范化核矩阵：",kn)

# 证明居中核矩阵，可以由居中化后的数据点乘得到
z1=x-np.mean(x,axis=0)
a1=z1[0,:]
a2=z1[1,:]
# 点积验证
a3=np.dot(a1,a2)
print("居中化后的数据x1,x2点乘：",a3)

# 证明规范核矩阵，可以由规范化后的数据点乘得到
# 首先对x1求其二范数
x1=x[0,:]
x1_norm=np.linalg.norm(x1, ord=2)
# 对x1规范化
b1=x1/x1_norm
# 同样的操作对x2
x2=x[1,:]
x2_norm=np.linalg.norm(x2, ord=2)
b2=x2/x2_norm
# 点积验证
b3=np.dot(b1,b2)
print("规范化后的数据x1,x2点乘",b3)