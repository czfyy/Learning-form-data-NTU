# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 22:47:45 2019

@author: czfyy
"""

import numpy as np
import matplotlib.pyplot as plt

def flip_coins(n):
    '''n枚硬币，每枚掷十次
    n:硬币数量
    '''
    #result:掷硬币结果，生成n行10列的ndarray，行数代表硬币数，列数代表掷十次的结果
    result = np.random.randint(0, 2, [n, 10])
    
    #正面朝上的数量
    heads = np.sum(result, axis=1)  #axis=1 是把列消掉，只剩行数； axis=0是把行消掉
    
    v1 = heads[0]
    vrand = heads[np.random.randint(0, n)]
    vmin = np.min(heads)
    
    return v1, vrand, vmin

#计算正面的次数
def total(x):
    s = 0
    for key,val in enumerate(x):
        s += key*val
    return s

def hist(n1, nrand, nmin, t):
    fig = plt.figure(figsize=(5,8))
    fig.clf()
    
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    
    plt.subplot(3,1,1)
    plt.bar(range(11),n1)
    plt.title('C1平均正面比例: %.4f' % (total(n1) / t))

    
    plt.subplot(3,1,2)
    plt.bar(range(11),nrand)
    plt.title('Crand平均正面比例: %.4f' % (total(nrand) / t))
    
    plt.subplot(3,1,3)
    plt.bar(range(11),nmin)
    plt.title('Cmin平均正面比例: %.4f' % (total(nmin) / t))
    
    fig.tight_layout()
    '''tight_layout紧凑布局，可自定义三个参数
    Pad:用于设置绘图区边缘与画布边缘的距离大小
    w_pad:用于设置绘图区之间的水平距离的大小
    H_pad:用于设置绘图区之间的垂直距离的大小
    '''
    plt.show()
    
def main():
    '''画直方图
    n:硬币数量
    m：重复m次实验
    t:某一枚被选中的硬币投掷总次数
    '''
    
    n = 1000
    m = 10000
    t = m*10
    
    #记录正面硬币得分0-10的次数，存入一个长度为11的列表，第i个元素表示得分为i的次数
    n1 = np.zeros(11)
    nrand = n1.copy()
    nmin = n1.copy()
    
    for i in range(m):
        v1, vrand, vmin = flip_coins(n)
        n1[v1] += 1
        nrand[vrand] += 1
        nmin[vmin] += 1
    
    print(n1)
    hist(n1, nrand, nmin, t)

if __name__ == '__main__':
    main()
        