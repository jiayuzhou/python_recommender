'''
This program computes and outputs the similar programs.
The results are prepared for Wook's team. 

Deadline is Jan 31st, 2014.

Created on Jan 29, 2014

@author: jiayu.zhou
'''

import numpy as np;
from scipy.sparse import coo_matrix;
from rs.data.daily_watchtime import DailyWatchTimeReader
from rs.utils.sparse_matrix import normalize_row;

if __name__ == '__main__':
    
    # define the data file. 
    filename = "../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    # load data from file and transform into a sparse matrix. 
    reader = DailyWatchTimeReader();
    fbdata = reader.read_file_with_minval(filename, 1, 1);  
    
    mat = coo_matrix((fbdata.data_val, (fbdata.data_row, fbdata.data_col)), \
                     shape = (fbdata.num_row, fbdata.num_col));
    # memo: if we do multiple days, we can use coo_matrix summation.  
    
    
    # normalize data per user. 
    mat = normalize_row(mat);
    
    
    
    program_mapping = fbdata.col_mapping; # from program id to row.
     
    program_inv_mapping = {y: x for x, y in program_mapping.items()}; # allows us to find program ID from matrix position.
    
    if not (len(program_mapping) == len(program_inv_mapping)):
        raise ValueError('Mapping inverse error!');
    
    program_num = len(program_mapping);
    
    cor_mat = np.zeros((program_num, program_num));
    
    for i in range(program_num):
        for j in range(program_num):
            if i > j:
                cor_mat[i][j] = mat.getcol(i) 
    
    pass;




