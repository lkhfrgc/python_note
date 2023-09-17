import numpy as np

"""
数组操作

更改形状----------------------------------------------------------------------------------
在对数组进行操作时，为了满足格式和计算的要求通常会改变其形状。

numpy.ndarray.shape表示数组的维度，返回一个元组，这个元组的长度就是维度的数目，即 ndim 属性(秩)。

numpy.ndarray.flat 将数组转换为一维的迭代器，可以用for访问数组每一个元素。

numpy.ndarray.flatten([order='C']) 将数组的副本转换为一维数组，并返回。
order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'k' -- 元素在内存中的出现顺序。(简记)
order：{'C / F，'A，K}，可选使用此索引顺序读取a的元素。'C'意味着以行大的C风格顺序对元素进行索引，
最后一个轴索引会更改F表示以列大的Fortran样式顺序索引元素，其中第一个索引变化最快，最后一个索引变化最快。
请注意，'C'和'F'选项不考虑基础数组的内存布局，仅引用轴索引的顺序.A'表示如果a为Fortran，
则以类似Fortran的索引顺序读取元素在内存中连续，否则类似C的顺序。
“ K”表示按照步序在内存中的顺序读取元素，但步幅为负时反转数据除外。默认情况下，使用Cindex顺序。
flatten()函数返回的是拷贝。

>>>>
numpy.ndarray.flat 和 numpy.ndarray.flatten 都是用于将数组展平的函数。

numpy.ndarray.flat 是一个一维迭代器，它共享了原始数组的内存空间。
这意味着对视图所做的修改会反映在原始数组上，反之亦然。它允许你以一维的方式访问和修改数组的元素。

numpy.ndarray.flatten 是一个类似于 numpy.ndarray.flat 的函数，但它返回的是一个一维数组，
而不是一个迭代器。这意味着它会创建原始数组的一个副本，不共享内存空间。因此，对返回的展平数组所做的修改不会反映在原始数组上。

在大多数情况下，使用 numpy.ndarray.flat 更为高效，因为它不需要创建副本并共享内存空间。
但是在某些情况下，如果你需要创建一个新的数组副本，那么可以使用 numpy.ndarray.flatten。
>>>>

numpy.ravel(a, order='C')
ravel()返回的是视图。
order=F 就是拷贝

numpy.reshape(a, newshape, order='C')
reshape()函数当参数newshape = [rows,-1]时，将根据行数自动确定列数。
reshape()函数当参数newshape = -1时，表示将数组降为一维。
返回的数组和原数组共享内存，仅仅是改变了形状

数组转置-------------------------------------------------------------------------
numpy.transpose(a, axes=None) 
Permute the dimensions of an array.

numpy.ndarray.T 
Same as self.transpose(), except that self is returned if self.ndim < 2.

-numpy.transpose(a, axes=None)是一个函数，可以通过指定axes参数来交换数组的不同轴。
而numpy.ndarray.T是一个属性，没有提供额外的参数来交换轴。
-当使用numpy.transpose(a, axes=None)时，输入的数组a必须是二维或更高维的。如果数组是一维的，直接返回原数组。
而numpy.ndarray.T可以用于任何维度的数组，返回的都是该数组的转置版本。
-numpy.transpose(a, axes=None)的输出是一个新的数组，与输入数组不共享内存空间。
而numpy.ndarray.T返回的是数组的转置视图，与原数组共享同一块内存空间。
-在性能方面，对于大型数组，使用numpy.ndarray.T通常比使用numpy.transpose(a, axes=None)更快，因为它不需要创建新的数组副本。


更改维度------------------------------------------------------------------------
当创建一个数组之后，还可以给它增加一个维度，这在矩阵计算中经常会用到。
numpy.newaxis用于创建一个新轴，将数组提升为更高维度。
很多工具包在进行计算时都会先判断输入数据的维度是否满足要求，
如果输入数据达不到指定的维度时，可以使用newaxis参数来增加一个维度。


numpy.squeeze(a, axis=None) 从数组的形状中删除单维度条目，即把shape中为1的维度去掉。
a表示输入的数组；
axis用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错；
在机器学习和深度学习中，通常算法的结果是可以表示向量的数组（即包含两对或以上的方括号形式[[]]），
如果直接利用这个数组进行画图可能显示界面为空（见后面的示例）。
我们可以利用squeeze()函数将表示向量的数组转换为秩为1的数组，
这样利用 matplotlib 库函数画图时，就可以正常的显示结果了。

数组组合--------------------------------------------------------------------------------------------
如果要将两份数据组合到一起，就需要拼接操作。

numpy.concatenate((a1, a2, ...), axis=0, out=None) 
Join a sequence of arrays along an existing axis.
连接沿现有轴的数组序列
原来x，y都是一维的，拼接后的结果也是一维的。
原来x，y都是二维的，拼接后的结果也是二维的。
x，y在原来的维度上进行拼接。

numpy.stack(arrays, axis=0, out=None)Join a sequence of arrays along a new axis.
沿着新的轴加入一系列数组（stack为增加维度的拼接）。

numpy.vstack(tup)
Stack arrays in sequence vertically (row wise).

numpy.hstack(tup)
Stack arrays in sequence horizontally (column wise).

hstack(),vstack()分别表示水平和竖直的拼接方式。在数据维度等于1时，比较特殊。
而当维度大于或等于2时，它们的作用相当于concatenate，用于在已有轴上进行操作。

数组拆分----------------------------------------------------------------------------------
numpy.split(ary, indices_or_sections, axis=0) 
Split an array into multiple sub-arrays as views （视图） into ary.
-array：输入的一维数组。
-indices_or_sections：表示分割的索引或段数。
如果是一个整数，那么该数组将会被等分成该整数指定的段数。
如果是一个整数数组，那么该数组将会按照指定的索引进行分割。
-axis（可选）：分割的轴，默认为0。
-返回一个列表，其中包含了分割后的子数组。

numpy.vsplit(ary, indices_or_sections) 
Split an array into multiple sub-arrays vertically (row-wise).
垂直切分是把数组按照高度切分

水平切分是把数组按照宽度切分。
numpy.hsplit(ary, indices_or_sections) 
Split an array into multiple sub-arrays horizontally (column-wise).

数组平铺------------------------------------------------------------------------------------
numpy.tile(A, reps) 
Construct an array by repeating A the number of times given by reps.
tile是瓷砖的意思，顾名思义，这个函数就是把数组像瓷砖一样铺展开来。
将原矩阵横向、纵向地整体复制。
-A 是输入数组。
-reps 是一个表示重复次数的整数或数组。
如果 reps 是一个整数，那么输入数组将在每个维度上重复该整数次。
如果 reps 是一个整数数组，那么该数组将会按照指定的索引进行重复。
(dim_1上的复制次数, dim_2...)

numpy.repeat(a, repeats, axis=None) 
-a 是输入的数组。
-repeats 是重复的次数，可以是单个整数或者长度与 a 相同的数组。
如果是一个整数，那么 a 在每个维度上都会被重复该整数次。
如果是一个数组，那么 a 会在每个维度上按照对应的元素进行重复。
-axis 是指定重复的轴。默认是 None，表示在所有轴上进行重复操作，即将数组展开成一维的向量。
如果指定为某个轴，那么操作就会在该轴上进行，返回的结果是形状在其余轴上保持不变的数组。


添加和删除元素---------------------------------------------------------------------------------------
numpy.append函数用于将一个或多个元素添加到NumPy数组的末尾。

numpy.append(arr, values, axis=None)

-arr：要添加元素的目标NumPy数组。
-values：要添加到数组末尾的元素或数组。可以是一个单独的数值、一维数组或者列表。
-axis（可选）：指定沿着哪个轴将values添加到数组中。
如果axis为None（默认值），则将values添加到数组的末尾，相当于沿着新维度进行扩展。
如果axis是一个整数，则将values沿着该轴添加到数组中。

返回一个新的NumPy数组，它包含原始数组和添加的元素(浅拷贝)


numpy.delete(arr, obj, axis=None)

-arr：输入的NumPy数组。
-obj：指定要删除的元素或切片，可以是整数、切片或列表。
-axis（可选）：指定要删除元素的轴，默认为None。
对于一维数组，该参数不起作用，因为只有一个轴。对于多维数组，该参数指定要删除元素的轴。

返回删除了指定元素的数组的副本（深拷贝）
"""
# 将 arr转换为2行的2维数组
arr = np.arange(10)
x = np.reshape(arr,(2,-1))
print(arr,'\n')
print(x,'\n')

# 堆叠数组a和数组b
a = np.arange(10).reshape([2, -1])
b = np.repeat(1, 10).reshape([2, -1])

x = np.concatenate((a,b))
y = np.concatenate((a,b), axis=1)
print(x,'\n')
print(y,'\n')

# 将 arr的2维数组按列输出
arr = np.array([[16, 17, 18, 19, 20],[11, 12, 13, 14, 15],[21, 22, 23, 24, 25],[31, 32, 33, 34, 35],[26, 27, 28, 29, 30]])

# np.apply_along_axis(lambda e:print(e),0, arr)
y = arr.flatten(order='F')
print(y)

# 给定两个随机数组A和B，验证它们是否相等

A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

print(A,'\n',B,'\n',np.allclose(A,B),'\n')

# 在给定的numpy数组中找到重复的条目（第二次出现以后），并将它们标记为True。第一次出现应为False。
np.random.seed(100)
a = np.random.randint(0, 5, 10)
print(a)
# [0 0 3 0 2 4 2 2 2 2]
b = np.full(10, True)
vals, counts = np.unique(a, return_index=True)
b[counts] = False
print(b)
# [False True False True False False True True]

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(20200605)
n = 2# 做某件事情的次数,这里是投两次硬币
p = 0.5#做某件事情成功的概率，在这里即投硬币为正面的概率
size = 50000
x = np.random.binomial(n, p, size)
'''或者使用binom.rvs
#使用binom.rvs(n, p, size=1)函数模拟一个二项随机变量,可视化地表现概率
y = stats.binom.rvs(n, p, size=size)#返回一个numpy.ndarray
'''
print(np.sum(x == 0) / size)  # 0.25154
print(np.sum(x == 1) / size)  # 0.49874
print(np.sum(x == 2) / size)  # 0.24972
matplotlib.rc("font",family='FangSong')
plt.hist(x, density=True)
plt.xlabel('随机变量：硬币为正面次数')
plt.ylabel('50000个样本中出现的次数')
plt.show()
#它返回一个列表，列表中每个元素表示随机变量中对应值的概率
s = stats.binom.pmf(range(n + 1), n, p)
print(np.around(s, 3))