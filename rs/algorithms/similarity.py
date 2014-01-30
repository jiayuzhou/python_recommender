'''
Similarity metrics. 

Created on Jan 29, 2014

@author: jiayu.zhou
'''

import math;
from itertools import izip;


def dot_product(v1, v2):
    '''
    Computer the inner product given two vectors in the form of list.  
    '''
    return sum(map(lambda x: x[0] * x[1], izip(v1, v2)))

def cosine_measure(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)