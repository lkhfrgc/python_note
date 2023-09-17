"""
解决中文乱码

DengXian
FangSong
KaiTi
LiSu
YouYuan
Adobe Fan Heiti Std
Adobe Fangsong Std
Adobe Heiti Std
Adobe Kaiti Std

import matplotlib
matplotlib.rc("font",family='YouYuan')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(20200605)
lam = 42 / 6# 平均值：平均每十分钟接到42/6次订票电话
size = 50000
x = np.random.poisson(lam, size)
'''或者
#模拟服从泊松分布的50000个随机变量
x = stats.poisson.rvs(lam,size=size)
'''

print(np.sum(x == 6) / size)  # 0.14988
plt.rc("font",family='YouYuan')
plt.hist(x)
plt.xlabel('随机变量：每十分钟接到订票电话的次数')
plt.ylabel('50000个样本中出现的次数')
plt.show()
#用poisson.pmf(k, mu)求对应分布的概率:概率质量函数 (PMF)
x = stats.poisson.pmf(6, lam)
print(x)  # 0.14900277967433773