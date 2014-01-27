'''
Created on Jan 24, 2014

@author: jiayu.zhou
'''

import numpy as np;
from visualization.histgram import histplot;


if __name__ == '__main__':
    
    data = np.random.randn(10000);
    histplot(data);