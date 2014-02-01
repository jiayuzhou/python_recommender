'''
Created on Jan 28, 2014

@author: jiayu.zhou
'''

from rs.data.generic_data import GenericData;
from rs.utils.sparse_matrix import normalize_row;
from scipy.sparse import coo_matrix;
import random

class FeedbackData(GenericData):
    '''
    A data structure for typical recommender system.  
    '''
    
    def __init__(self, data_row, data_col, data_val, num_row, num_col,\
                  row_mapping = None, col_mapping = None, meta_data = None):
        '''
        data tuples: (data_row[i], data_col[i], data_val[i])
        number of row/col: num_row, num_col;
        name of rows/cols in a dictionary format: row_mapping, col_mapping
        '''
        # check the consistency of the data elements. 
        if (not len(data_row) == len(data_col)) or (not len(data_row) == len(data_val)):
            print;
        
        # the following things form a tuple (data_row[i], data_col[i], data_val[i]).
        # later on a data_row can construct a coo sparse matrix.  
        self.data_row = data_row; 
        self.data_col = data_col;
        self.data_val = data_val;
        self.num_row  = num_row;
        self.num_col  = num_col;
        
        self.row_mapping  = row_mapping;
        self.col_mapping = col_mapping;
        self.meta = meta_data;
    
    def __str__(self):
        return 'Row#:' + str(self.num_row) + ' Col#:' + str(self.num_col) + ' Element:'+ str(len(self.data_val));
     
    def get_sparse_matrix(self):
        mat = coo_matrix((self.data_val, (self.data_row, self.data_col)), \
                     shape = (self.num_row, self.num_col));
        return mat;
        
    def normalize_row(self):
        '''
        Perform a normalization on the row. This will modify the row/col/val.
        The method first constructs a coo sparse matrix and then convert to lil 
        matrix for normalization. The normalized matrix is then converted back to 
        coo matrix and row/col/val are reset to the values in the converted coo matrix. 
        
        NOTE: when we have (row, col, 0.1) and (row, col, 0.2), and we will have 
              (row, col, 0.3), because coo to lil transformation.  
        '''
        mat = coo_matrix((self.data_val, (self.data_row, self.data_col)), \
                     shape = (self.num_row, self.num_col));
        mat = normalize_row(mat);
        mat = mat.tocoo();
        
        self.data_val = mat.data.tolist();
        self.data_row = mat.row.tolist();
        self.data_col = mat.col.tolist();
        
        
        
    def subsample_row(self, user_number):
        '''
        randomly sub-sample the data of a given number of users. This will modify 
        the row/col/val. 
        The method first constructs a coo sparse matrix and then convert to csr matrix 
        for row slicing. And then the csr matrix is converted back.
        '''
        if user_number > self.num_row:
            user_number = self.num_row;
        
        # construct sparse matrix using coo.
        mat = coo_matrix((self.data_val, (self.data_row, self.data_col)), \
                     shape = (self.num_row, self.num_col));
                     
        # convert to csr for fast row slicing. 
        mat = mat.tocsr();
        
        # generate random sample index
        idx = range(mat.shape[0]);
        random.shuffle(idx);             # random permutation.
        selidx = idx[:user_number];      # take random rows.
        mat = mat[selidx, :];            # slice the matrix. 
        mat = mat.tocoo();               # convert it back. 
        
        data_val = mat.data.tolist();
        data_row = mat.row.tolist();
        data_col = mat.col.tolist();
        
        newdata = FeedbackData(data_row, data_col, data_val, user_number, self.num_col,\
                  self.row_mapping, self.col_mapping, self.meta); # generate new data 
                  
        return [newdata, selidx];
        
    def split(self, percentage):
        '''
        get a random splitting of data with a specified proportion. 
        '''
        pass;
    
    def fold(self, fold_num, total_fold):
        '''
        get the n-th fold of the n fold data.
        '''
        pass;
    
    
    
    