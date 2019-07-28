# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:14:30 2019

@author: czfyy
"""

import numpy as np
import matplotlib.pyplot as plt


def zero_one_error(s, y):
    if s * y > 0:
        return 0
    else:
        return 1

def squared_error(s, y):
    return (s - y)**2

def cross_entropy_error(s, y):
    return np.log(1 + np.exp(0 - y*s))

def err_plt():
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

    s = np.linspace(-2, 2, 500)
    
    zero_one_error_positive = [zero_one_error(i,1) for i in s]
    zero_one_error_negative = [zero_one_error(i,-1) for i in s]
    
    squared_error_positive = [squared_error(i,1) for i in s]
    squared_error_negative = [squared_error(i,-1) for i in s]
    
    cross_entropy_error_positive = [cross_entropy_error(i,1) for i in s]
    cross_entropy_error_negative = [cross_entropy_error(i,-1) for i in s]
    
    fig = plt.figure(figsize=(5,8))
    fig.clf()
    
    plt.subplot(2,1,1)
    plt.plot(s, zero_one_error_positive, label='zero_one_error')
    plt.plot(s, squared_error_positive, label='squared_error')
    plt.plot(s, cross_entropy_error_positive, linestyle='--', label='cross_entropy_error')
    plt.plot(s, cross_entropy_error_positive/np.log(2), label='scaled_cross_entropy_error')
    plt.title('y=1')
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(s, zero_one_error_negative, label='zero_one_error')
    plt.plot(s, squared_error_negative, label='squared_error')
    plt.plot(s, cross_entropy_error_negative, linestyle='--',label='cross_entropy_error')
    plt.plot(s, cross_entropy_error_negative/np.log(2), label='scaled_cross_entropy_error')
    plt.title('y=-1')
    plt.legend()
    
    fig.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    err_plt()