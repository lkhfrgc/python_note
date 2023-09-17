import numpy as np

"""
1. 依据现有数据来创建 ndarray---------------------------------------
（a）通过array()函数进行创建
a = np.array([0, 1, 2, 3, 4])

（b）
asarray()函数进行创建：
使用原存储空间的数据，不会另行创建
array()：
创建新的数组，把原数据拷贝到其中

（c）通过fromfunction()函数进行创建
给函数绘图的时候可能会用到fromfunction()，
该函数可从函数中创建数组
def f(x, y):
    return 10 * x + y
x = np.fromfunction(f, (5, 4), dtype=int)
print(x)
# [[ 0  1  2  3]
#  [10 11 12 13]
#  [20 21 22 23]
#  [30 31 32 33]
#  [40 41 42 43]]


2. 依据 ones 和 zeros 填充方式---------------------------------------
（a）零数组
def zeros(shape, dtype=None, order='C'):
def zeros_like(a, dtype=None, order='K', subok=True, shape=None)
（b）1数组
def ones(shape, dtype=None, order='C'):
def ones_like(a, dtype=None, order='K', subok=True, shape=None)
（c）空数组
def empty(shape, dtype=None, order='C'): 
def empty_like(prototype, dtype=None, order='K', subok=True, shape=None):
（d）单位数组
eye()函数：返回一个对角线上为1，其它地方为零的单位数组。
identity()函数：返回一个方的单位数组。
def eye(N, M=None, k=0, dtype=float, order='C'):
def identity(n, dtype=None):
（e）对角数组
diag()函数：提取对角线或构造对角数组。
def diag(v, k=0)
（f）常数数组
full()函数：返回一个常数数组。
full_like()函数：返回与给定数组具有相同形状和类型的常数数组。
def full(shape, fill_value, dtype=None, order='C'):
def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):


3. 利用数值范围来创建ndarray---------------------------------------
arange()函数：返回给定间隔内的均匀间隔的值。
linspace()函数：返回指定间隔内的等间隔数字。
logspace()函数：返回数以对数刻度均匀分布。
numpy.random.rand()/normal()/random()/... 传入数组形状，返回随机数组成的数组

def arange([start,] stop[, step,], dtype=None): 
def linspace(start, stop, num=50, endpoint=True, retstep=False, 
             dtype=None, axis=0):
def logspace(start, stop, num=50, endpoint=True, base=10.0, 
             dtype=None, axis=0):



4. 结构数组的创建---------------------------------------
（a）利用字典来定义结构
personType = np.dtype({
    'names': ['name', 'age', 'weight'],
    'formats': ['U30', 'i8', 'f8']})

a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
print(a, type(a))
# [('Liming', 24, 63.9) ('Mike', 15, 67. ) ('Jan', 34, 45.8)]
# <class 'numpy.ndarray'>

（b）利用包含多个元组的列表来定义结构
personType = np.dtype([('name', 'U30'), ('age', 'i8'), ('weight', 'f8')])
a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
print(a, type(a))


数组的属性---------------------------------------
在使用 numpy 时，你会想知道数组的某些信息。很幸运，在这个包里边包含了很多便捷的方法，可以给你想要的信息。

numpy.ndarray.ndim用于返回数组的维数（轴的个数）也称为秩，一维数组的秩为 1，二维数组的秩为 2，以此类推。
numpy.ndarray.shape表示数组的维度，返回一个元组，这个元组的长度就是维度的数目，即 ndim 属性(秩)。
numpy.ndarray.size数组中所有元素的总量，相当于数组的shape中所有元素的乘积，例如矩阵的元素总量为行与列的乘积。
numpy.ndarray.dtype ndarray 对象的元素类型。
numpy.ndarray.itemsize以字节的形式返回数组中每一个元素的大小。
"""


arr = np.arange(10)
print(arr)
arr = np.full([3,3],True,dtype=bool)
print(arr)
arr = np.arange(10,50)
print(arr)
arr = np.random.random((3,3))
print(arr)
arr = np.ones((10,10))
arr[1:-1,1:-1] = 0
print(arr)