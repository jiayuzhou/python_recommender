'''
Produce indices for data splitting and folding. 

Created on Feb 1, 2014

@author: jiayu.zhou
'''

import random;

def sample_num(data_size, sel_num):
    '''
    randomly sub-sample the data of a given number of elements. 
    
    Parameters
    ----------
    data_size: the size of the data set.
    sel_num: the number of selected elements. 
    
    Returns
    ----------
    sel_idx: a list of indices with the given number. 
    '''
    if sel_num > data_size:
        sel_num = data_size;
    if sel_num < 0:
        raise ValueError('The number of selected items cannot be negative. ');
        
    # generate random sample index
    idx = range(data_size);  
    random.shuffle(idx);             # random permutation.
    sel_idx = idx[:sel_num];          # take random rows.
        
    return sel_idx

def split(data_size, percentage):
    '''
    produce a random splitting index pair 
    
    Parameters
    ----------
    data_size: the size of the data set.
    percentage: the percentage of data split. 
    
    Returns
    ----------
    out: a list [sel_idx, sel_idx_comp]
    sel_idx:
    sel_idx_comp:
    
    '''
    if percentage >= 1 or percentage <=0:
        raise ValueError('percentage should be in the open range of (0, 1).')
    
    sel_row_num = int(round(data_size * percentage));
    sel_row_num = max(min(sel_row_num, data_size), 1); # range protection [1, self.num_row]
    
    # get the selected list of the split. 
    sel_idx      = sample_num(data_size, sel_row_num);
    # get the compliment of the split. 
    sel_idx_comp = list(set(range(data_size)) - set(sel_idx));
    
    return [sel_idx, sel_idx_comp]; 
    
def fold(data_size, fold_num, total_fold):
    '''
    get the index of a fold of n-fold.  
    This is deterministic.
    
    Parameters
    ----------
    data_size:  number of 
    fold_num:   the 0-based fold index [0..total_fold-1] 
    total_fold: 
    
    Returns
    ----------
    out. the selected 
    '''
    
    if fold_num >= total_fold:
        raise ValueError("Fold number should be <= total_fold - 1");
    return range(data_size)[fold_num::total_fold];
    
    pass;