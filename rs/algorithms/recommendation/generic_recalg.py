'''
Created on Jan 29, 2014

@author: jiayu.zhou
'''

from rs.algorithms.Alg import Alg;

class CFAlg(Alg):
    '''
    Generic collaborative filtering algorithm. The algorithms in the interface 
    should have a train method that takes an input of the type:
        rs.data.recdata.FeedbackData
    And also a predict function takes a list of positions (that previous unknown).
    '''

    def train(self, feedback_data):
        '''
        Train the model with specified feedback_data. 
        WARNING: calling this function will erase previous training information. 
        
        Parameters
        ----------
        Feedback data structure in rs.data.recdata.FeedbackData
         
        Returns
        ----------
        no return.
        '''
        
        raise NotImplementedError("Interface method.");
    
    def predict(self, row_idx_arr, col_idx_arr):
        '''
        Prediction elements in specified locations. The index is 0-based. 
        The prediction at (row_idx_arr(i), col_idx_arr(j)) is U[i, :] * V[:, col].
        
        Parameters
        __________
        row_idx_arr : a list of 'row' part of the locations.
        col_idx_arr : a list of 'col' part of the locations.  
        
        Returns
        __________
        return a list of results (predicted values) at specified locations.   
        '''
        
        raise NotImplementedError("Interface method.");
    
    def predict_row(self, row_idx, col_idx_arr):
        '''
        Predict elements in specific locations for one row (user). The index is 0-based. 
        
        Parameters
        ----------
        @param row_idx:     the index or the row (user), 0-based. 
        @param col_idx_arr: the indices for items. 
        
        Returns
        ----------
        @return: return a list of results (predicted values) at specified locations. 
        '''
    
    def unique_str(self):
        '''
        The unique string should include the algorithm name and parameter values. 
        '''
        pass;