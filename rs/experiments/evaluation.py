'''
Provides a list of evaluation metrics.

Created on Feb 3, 2014

@author: jiayu.zhou
'''

from itertools import izip;

def rmse(list1, list2):
    '''
    Return the root mean squared error.  
    sqrt(sum_i ((x[i] - y[i])**2))
    '''
    if not len(list1) == len(list2):
        raise ValueError('Dimension not match.');
    
    return (sum( (x - y) ** 2 for x, y in izip(list1, list2)) / len(list1) ) ** 0.5;


def hit_prec(hit_status):
    '''
    Evaluate the precision of hit. The input is the hit status (>0 means hit) of 
    a list of recommended items (items ranked top by the recommendation algorithm). 
    '''
    
    return sum( x > 0 for x in hit_status ) / float(len(hit_status));

def precision_itemlist(pred_list, hit_item_list):
    a = set(pred_list);
    b = set(hit_item_list);
    # nan case. 
    if len(a) == 0:
        return 0;
    
    return len(a & b)/float(len(a))

def recall_itemlist(pred_list, hit_item_list):
    a = set(pred_list);
    b = set(hit_item_list);
    if len(b) == 0:
        return 0;
    
    return len(a & b)/float(len(b))
    




