'''
Created on Jan 28, 2014

@author: jiayu.zhou
'''

from rs.data.generic_data import GenericData;

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
        self.item_mapping = col_mapping;
        self.meta = meta_data;
    
    def __str__(self):
        return 'Row#:' + str(self.num_row) + ' Col#:' + str(self.num_col) + ' Element:'+ str(len(self.data_val)); 
    
    def subsample_row(self, user_number):
        '''
        subsample the data of a set of users.  
        '''
        pass;
        
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
    
    
    
    