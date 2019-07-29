# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:23:56 2019

@author: czfyy
"""

import matplotlib.pyplot as plt
import numpy as np

def problem5(N, dvc):
    return (N**dvc) + 1

def problem6(N, dvc):
    return (np.exp(1)*N / dvc)**dvc

def main():
    n = np.arange(1, 50, 0.1)
    m_H1 = [problem5(i, 2) for i in n]
    m_H2 = [problem6(i, 2) for i in n]
    
    m_H11 = [problem5(i, 5) for i in n]
    m_H22 = [problem6(i, 5) for i in n]
    
    fig = plt.figure(figsize=(5,8))
    fig.clf()
        
    plt.subplot(2,1,1)
    plt.plot(n, m_H1, label='problem5')
    plt.plot(n, m_H2, label='problem6')
    plt.title('dvc=2')
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(n, m_H11, label='problem5')
    plt.plot(n, m_H22, label='problem6')
    plt.title('dvc=5')
    plt.legend()
    
    fig.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()