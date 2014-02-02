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
        '''
        Construct a sparse matrix (coo_matrix) from current Feedback data content.  
        '''
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
        mat = self.get_sparse_matrix();
        mat = normalize_row(mat); # normalize each row. 
        mat = mat.tocoo();
        
        self.data_val = mat.data.tolist();
        self.data_row = mat.row.tolist();
        self.data_col = mat.col.tolist();
        
    def subdata(self, selidx):
        '''
        select a set of rows (users) to form a new dataset.  
        
        Parameters
        ----------
        selidx: a list of row indices. The row index set can have duplicated entries.  
        
        Returns
        ----------
        out: a new FeedbackData instance whose rows are given in selidx 
             in the original set. 
        '''
        # check if the indices are good. 
        if max(selidx) >= self.num_col or min(selidx) < 0:
            raise ValueError('Found invalid element in the index set!');
        
        # construct sparse matrix using coo.
        mat = self.get_sparse_matrix();
                     
        # convert to csr for fast row slicing. 
        mat = mat.tocsr();
        
        # sub-sampling data matrix.
        mat = mat[selidx, :];            # slice the matrix. 
        mat = mat.tocoo();               # convert it back. 
        
        # recompute the row mapping. 
        inv_mapping = {yy : xx for xx, yy in self.row_mapping.iteritems()};
        row_mapping = { inv_mapping[xx]: ii for ii, xx in enumerate(selidx) };
        
        data_val = mat.data.tolist();
        data_row = mat.row.tolist();
        data_col = mat.col.tolist();
        
        newdata = FeedbackData(data_row, data_col, data_val, len(selidx), self.num_col,\
                  row_mapping, self.col_mapping, self.meta); # generate new data 
                  
        return newdata;
        
    def subsample_row(self, sel_row_num):
        '''
        randomly sub-sample the data of a given number of users. This will modify 
        the row/col/val. 
        The method first constructs a coo sparse matrix and then convert to csr matrix 
        for row slicing. And then the csr matrix is converted back.
        
        Parameters
        ----------
        
        Returns
        ----------
        out: a list of two components [sample_data, selidx]
        sample_data: 
        selidx
        '''
        if sel_row_num > self.num_row:
            sel_row_num = self.num_row;
        
        # generate random sample index
        idx = range(len(self.row_mapping));  
        random.shuffle(idx);                 # random permutation.
        selidx = idx[:sel_row_num];          # take random rows.
        
        sample_data = self.subdata(selidx);
        return [sample_data, selidx];
        
    def split(self, percentage):
        '''
        get a random splitting of data with a specified proportion of rows. 
        NOTE: it is recommended to use subdata method in get deterministic splits. 
        
        Parameters
        ----------
        percentage: the percentage of data split. 
        
        Returns
        ----------
        out: a list [data_split, data_split_comp, selidx_split, selidx_split_comp]
        data_split: a Feedback data of percentage, whose index (in the full data set) 
                    is given in selidx_split
        data_split_comp: a Feedback data of 1- percentage, whose index is given 
                         in data_split_comp. This is the complement part of data_split. 
        selidx_split: the index of rows in data_split.
        selidx_split_comp: the index of rows in data_split_comp. 
        '''
        
        if percentage >= 1 or percentage <=0:
            raise ValueError('percentage should be in the open range of (0, 1).')
        
        sel_row_num = int(round(self.num_row * percentage));
        sel_row_num = max(min(sel_row_num, self.num_row), 1); # range protection [1, self.num_row]

        # obtain a random part of sub data set of number sel_row_num.
        [data_split, selidx_split] = self.subsample_row(sel_row_num);
        
        # get the compliment of the split. 
        selidx_split_comp  = list(set(range(self.num_row)) - set(selidx_split));
        data_split_comp = self.subdata(selidx_split_comp);
        
        return [data_split, data_split_comp, selidx_split, selidx_split_comp];
    
    def fold(self, fold_num, total_fold):
        '''
        get the n-th fold of the n-fold data.
        
        Parameters
        ----------
        fold_num:
        total_fold:
        
        Returns
        ----------
        out: [fold_data, selidx]
        fold_data: 
        selidx
        '''
        pass;
    
    
    
    