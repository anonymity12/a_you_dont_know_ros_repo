# you dont know ros

[toc]

a repo for learn a particle filtering

# current stage

## stage 1 done 

![stage1](./img/stage1done.jpg)

now we know exactly where the robot is.

we did this by getting the average of all particles x and y

- search in ref_code.py for: **stage 1 done** . you can find the changes as follow:

![img2](./img/stage1change.jpg)

# useful res

## links:

https://github.com/mjl/particle_filter_demo
https://github.com/mit-racecar/particle_filter
https://github.com/leimao/Particle_Filter

## videos:

https://www.youtube.com/embed/7Z9fEpJOJdc
https://www.youtube.com/embed/TKCyAz063Yc	

### backup:

https://www.bilibili.com/video/av92849889/

## pdfs:

particle filtering docs:

http://web.mit.edu/16.412j/www/html/Advanced%20lectures/Slides/Hsaio_plinval_miller_ParticleFiltersPrint.pdf

https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf

## ref code:

cp from 

http://ros-developer.com/2019/04/10/parcticle-filter-explained-with-python-code-from-scratch/

## about python

### note!!

>you can find all the related stuff for how to use `numpy` in subfolder: u_learn_numpy_here

### Numpy-np.random.normal()正态分布

https://www.cnblogs.com/cpg123/p/11779117.html

正态分布（又称高斯分布）的概率密度函数


 
numpy中

numpy.random.normal(loc=0.0, scale=1.0, size=None) 

参数的意义为：

　　loc:float

　　概率分布的均值，对应着整个分布的中心center

　　scale:float

　　概率分布的标准差，对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高

 　 size:int or tuple of ints

　　输出的shape，默认为None，只输出一个值

我们更经常会用到np.random.randn(size)所谓标准正态分布（μ=0, σ=1），对应于np.random.normal(loc=0, scale=1, size)



### numpy.random.pareto

https://www.osgeo.cn/numpy/reference/generated/numpy.random.pareto.html


numpy.random.pareto(a, size=None)


从指定形状的Pareto II或Lomax分布中提取样品。

Lomax或Pareto II分布是一个移位的Pareto分布。经典的帕累托分布可以从lomax分布中通过加1和乘以尺度参数得到。 m （见注释）。Lomax分布的最小值是零，而对于经典的Pareto分布，它是零。 mu ，其中标准帕累托分布有位置 mu = 1 . Lomax也可以被视为广义帕累托分布的简化版本（在Scipy中可用），比例设置为1，位置设置为零。

帕累托分布必须大于零，并且在上面是无界的。它也被称为“80-20法则”。在这个分布中，80%的权重在最小的20%范围内，而其他20%则填充剩余的80%范围。

参数: 
a : 浮点数或类似浮点数的数组
分布的形状。应大于零。

size : int或int的元组，可选
输出形状。如果给定的形状是，例如， (m, n, k) 然后 m * n * k 取样。如果尺寸是 None （默认），如果 a 是标量。否则， np.array(a).size 取样。

返回: 
out : ndarray或scalar
从参数化帕累托分布中提取样本。

参见
scipy.stats.lomax
概率密度函数、分布或累积密度函数等。
scipy.stats.genpareto
概率密度函数、分布或累积密度函数等。
笔记

帕累托分布的概率密度是

p（x）=frac am^a x ^ a+1

在哪里？ a 是形状和 m 规模。

帕累托分布，以意大利经济学家维尔弗雷多·帕累托的名字命名，是一种适用于许多现实世界问题的幂律概率分布。在经济学领域之外，它通常被称为布拉德福德分布。帕累托发展了分布来描述经济中财富的分布。它还可以用于保险、网页访问统计、油田规模和许多其他问题，包括SourceForge项目的下载频率。 [1]. 它是所谓的“肥尾”分布之一。


### numpy linalg模块 

https://www.cnblogs.com/xieshengsen/p/6836430.html


