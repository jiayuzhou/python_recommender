'''
Produce indices for data splitting and folding. 

Created on Feb 1, 2014

@author: jiayu.zhou
'''

import random;

def sample_num(data_size, sel_num):
    '''
    
    '''
    if sel_num > data_size:
            sel_num = data_size;
        
    # generate random sample index
    idx = range(len(data_size));  
    random.shuffle(idx);                 # random permutation.
    sel_idx = idx[:sel_num];          # take random rows.
        
    return sel_idx

def split(data_size, percentage):
    '''
    produce a random splitting index pair 
    
    Parameters
    ----------
        
    Returns
    ----------
      
    '''
    if percentage >= 1 or percentage <=0:
        raise ValueError('percentage should be in the open range of (0, 1).')
    
    sel_row_num = int(round(data_size * percentage));
    sel_row_num = max(min(sel_row_num, data_size), 1); # range protection [1, self.num_row]
    
    
def fold(data_size, fold_num, total_fold):
    pass;