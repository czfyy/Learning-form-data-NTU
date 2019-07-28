# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:55:20 2019

@author: czfyy
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def generate_data(n, d, w0, seed):
    '''产生目标函数和数据集
    n:数据点个数，取值范围[-10, 10]
    d:维度
    seed:随机种子,任取
    注：w的值取在[-1，1]
    '''
    
    #设置随机数
    rnd = np.random.RandomState(seed)
    #随机产生f: target function (包括w0) , w保留5位小数
    w = np.round(rnd.uniform(-1, 1, d+1), 5)
    w[0] = w0                                         #将w0赋值成给定值
    print('actual w =', w)
    print()
    
    #产生数据,保留两位小数
    datax0 = np.zeros(n) + 1                               #根据公式，x0=1 恒成立
    datax = np.round(rnd.uniform(-10, 10, [d,n]), 2)       #产生一个d*n 的数列，每一列代表一个元素坐标，例如：三维数组就有三列，1000个数据点就有1000行
    data = np.vstack((datax0, datax))                      #将x0与x1...垂直拼接起来，正好符合公式要求
    #print(data)
    #print()
    
    #标记数据y值，将f>0数据标为+1
    label = np.ones(n)
    for i in range(n):
        if (w.dot(data[:,i])) <= 0:
            label[i] = -1
    
    label[: n//10] *= -1  #让前10%数据误分
    
    print('actual label =', label)
    print('------------------------------------------------')
    return w,data,label

def Pocket_PLA(data, label, step, flag=False):
    '''感知器算法
    flag: 默认False,  w从0开始；
         flag=True时: 令w=data[:,N],相当于从一个随机数开始
    '''
    if flag:
        N = np.random.randint(0, len(label))
        w = data[:,N]
    else:
        w = np.zeros(len(data[:,0]))   #产生个零向量，长度与数据点个数相同
        
    w = calculation(data, label, w, step)
    
    return w

def calculation(data, label, w, T):
    w_temp = np.zeros(len(w))
    step = 0
    error = count(data, label, w)
    run = True
    
    while run:                                         #循环
        for key,val in enumerate(label):                  #遍历所有数据
            print(key) 
            if w.dot(data[:,key])*val <= 0:       #遇到错误分类的点使：wTxy<0, 更新一次 w，
                w_temp = w + val * data[:,key]  
                error_new = count(data, label, w_temp)
                if error_new < error:
                    w = np.copy(w_temp)
                    step += 1
                    print('step =', step)
                    if step == T-1:
                        run = False
                        break
                        #break
            
    print('PLA label =  ', judge(data, label, w)) #当完整循环一遍所有数据后，通过judge函数判断是否还有错误数据
    print('steps =', step)                           #如果分类全部正确，judge将返回一个正确的label,此时就打印并退出循环
    print((judge(data, label, w) == label).all())
    print('PLA w =', w)
    print('------------------------------------------------')
            
    return w

def judge(x, y, w):
    '''判断yn是否正确
    label: yn值,     label是一个行向量,其包含全部数据点的yn值,索引值（index）与data是完全对应的
                     这样就可以直接计算向量形式的 wTxy 了
    '''
    flag = w.dot(x)*y < 0                #这里用的w是hypothesis， 但label是完全正确的
    
    if True not in flag:                #如果 wTxy<0都不成立（即 wTxy>0 恒成立），返回y，此时w就近似等于f
        test_label = y.copy()
        #print(id(test_label), id(y))
        #print(test_label)
        return test_label

def count(data, label, w):
    num = np.sum(w.dot(data)*label <= 0)
    return num

def plot_PLA(data, label, w_f, w_g):
    '''绘制数据点和分类器
    
    '''
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

    
    if (len(data[:,0]) - 1) == 2: #二维
        plt.clf()
        
        xf1 = np.arange(-12, 12, 0.1)
        xf2 = (w_f[0] + w_f[1]*xf1) / (-w_f[2])
        
        xg1 = xf1.copy()
        xg2 = (w_g[0] + w_g[1]*xg1) / (-w_g[2])
        
        plt.plot(xf1, xf2, 'r', label='target function', linestyle='-.')  #target function
        plt.plot(xg1, xg2, 'g', label='hypothesis')                       #hypothesis
        
        point1 = data[1:3,np.where(label>0)]                      #np.where可以返回label>0的点的位置，并通过该ndarray切片生成一组wTx>0的点的序列
        plt.scatter(point1[0,:],point1[1,:], c='b', marker='^')   #data[1:3,]3取不到！表示data的第二行和第三行！
        
        point2 = data[1:3,np.where(label<0)]                      #取出label<0的点，再作图
        plt.scatter(point2[0,:],point2[1,:], c='r')
        
        plt.axis([-12, 13, -12, 13])
        plt.legend(loc="upper right")
        plt.show()
        
    if (len(data[:,0]) - 1) == 3:#三维
        fig = plt.figure()
        fig.clf()
        ax = Axes3D(fig)
        
        xf1 = np.arange(-10, 10, 0.1)
        xf2 = xf1.copy()
        XF1,XF2 = np.meshgrid(xf1, xf2)                             #绘制三维图像要先产生网格
        XF3 = (w_f[0] + w_f[1]*XF1 + w_f[2]*XF2) / (-w_f[3])
        ax.plot_surface(XF1, XF2, XF3, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        xg1 = xf1.copy()
        xg2 = xf1.copy()
        XG1,XG2 = np.meshgrid(xg1, xg2)
        XG3 = (w_g[0] + w_g[1]*XG1 + w_g[2]*XG2) / (-w_g[3])
        ax.plot_surface(XG1, XG2, XG3, rstride=1, cstride=1, cmap=cm.summer, linewidth=0, antialiased=False)
        
        point1 = data[1:4,np.where(label>0)]
        ax.scatter(point1[0,:],point1[1,:],point1[2,:], c='k', marker='^')
        
        point2 = data[1:4,np.where(label<0)]
        ax.scatter(point2[0,:],point2[1,:],point2[2,:], c='r')
        
        plt.axis([-10, 10, -10, 10])
        ax.set_zlim(-20, 20)
        plt.show()

def main():
    num = 100
    dimension = 2
    w0 = 1.5
    seed = 10
    
    w_f, data, label = generate_data(num, dimension, w0, seed)
    
    T = 1000
    start = time.clock()
    w_g = Pocket_PLA(data, label, T)
    end = time.clock()
    
    print('actual w =', w_f)
    print(data.dtype, label.dtype, w_f.dtype)
    print('time:', end-start, 'seconds')
    print()
    
    plot_PLA(data, label, w_f, w_g)

if __name__ == '__main__':
    main()
