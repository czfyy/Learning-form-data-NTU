# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:57:19 2019

@author: czfyy
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


from Exercise4 import generate_data, PLA, judge, plot_PLA



def calculation(data, label, w):
    '''根据题干重写w更新公式和判断条件
    增加参数η用n表示
    '''
    step = 0
    n = 1     #η
    run = True
    
    while run:                                        #循环
        for key,val in enumerate(label):                  #遍历所有数据
            print(key)
            if w.dot(data[:,key])*val <= 1:       #遇到错误分类的点使：wTxy<0, 更新一次 w，
                w += n * (val - w.dot(data[:,key])) * data[:,key]            
                step += 1
                print('step =', step)
            
        if (judge(data, label, w) == label).all():        #.all()表示元素全部相同时返回True
            print('PLA label =  ', judge(data, label, w)) #当完整循环一遍所有数据后，通过judge函数判断是否还有错误数据
            print('steps =', step)                           #如果分类全部正确，judge将返回一个正确的label,此时就打印并退出循环
            print((judge(data, label, w) == label).all())
            print('PLA w =', w)
            print('------------------------------------------------')
            run = False
            break
    return w


def main():
    num = 1000
    dimension = 2
    w0 = 1.5
    seed = 10
    w_f, data, label = generate_data(num, dimension, w0, seed)
    
    start = time.clock()
    w_g = PLA(data, label)
    end = time.clock()
    
    print('actual w =', w_f)
    print(data.dtype, label.dtype, w_f.dtype)
    print('time:', end-start, 'seconds')
    print()
    
    plot_PLA(data, label, w_f, w_g)

if __name__ == '__main__':
    main()
