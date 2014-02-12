'''
Produce indices for data splitting and folding. 

Created on Feb 1, 2014

@author: jiayu.zhou
'''

import random;
import numpy as np;

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
    data_size:  number of samples. 
    fold_num:   the 0-based fold index [0..total_fold-1] 
    total_fold: the total number the (n) of n-fold.
    
    Returns
    ----------
    out. the selected index in this fold. 
    '''
    
    if fold_num >= total_fold:
        raise ValueError("Fold number should be <= total_fold - 1");
    return range(data_size)[fold_num::total_fold];

def leave_k_out(feedback_data, leave_k_out):
    '''
    leave k items out for each user. This code compute the items to be left out.  
    
    Parameters
    ----------
    @param feedback_data: the feedback data to be performed leave k out.
    @param leave_k_out:   the number of items to be left out. 
    
    Returns
    ---------- 
    @return leave_ind: an indexed dictionary, indices of the k left for each row. 
    '''
    
    leave_ind = {};
    
    smat = feedback_data.get_sparse_matrix().tocsr();
    
    # compute the non-zero elements for each row and randomly pick k out. 
    for i in range(smat.shape[0]):
        
        col_pos_arr = np.nonzero(smat[i, :])[1];
        # for each row slice compute the nnz. 
        nnzi = len(col_pos_arr);
        
        # safeguard.
        kk = max(0, min(nnzi - 1, leave_k_out)); # at least one element.
        
        idx = range(nnzi);  
        random.shuffle(idx);   # random permutation.
        sel_idx = idx[:kk];    # take random rows.
        
        #leave_ind[i] = sel_idx;
        leave_ind[i] = set(col_pos_arr[sel_idx]);
        
    return leave_ind;

