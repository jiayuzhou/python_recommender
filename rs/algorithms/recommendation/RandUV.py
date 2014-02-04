'''
Created on Jan 29, 2014

@author: jiayu.zhou
'''
import numpy as np;
from rs.algorithms.recommendation.generic_recalg import CFAlg;
from rs.utils.log import Logger; 

# an encapsulated logger.  
log = lambda message: Logger.Log(RandUV.ALG_NAME + ':'+message, Logger.MSG_CATEGORY_ALGO);

class RandUV(CFAlg):
    '''
    A random guess recommender (demo).
    '''
    ALG_NAME = 'RANDOM ALGO';

    def __init__(self, latent_factor = 5, verbose = False):
        '''
        Constructor
        '''
        # initialize parameters. 
        self.latent_factor = latent_factor;
        log('dummy algorithm instance created: latent factor ' + str(self.latent_factor));
        
        self.verbose = verbose;
            
    def unique_str(self):
        return RandUV.ALG_NAME + '_k_' + str(self.latent_factor);
    
    def train(self, feedback_data):
        '''
        Train the model with specified feedback_data. 
        WARNING: calling this function will erase previous training information. 
        
        Parameters
        ----------
        feedback_data: feedback data structure in rs.data.recdata.FeedbackData
         
        Returns
        ----------
        no return.
        '''
        if self.verbose:
            log('training dummy algorithm.');
        
        m = feedback_data.num_row;
        n = feedback_data.num_col;  
        r = self.latent_factor;
        
        self.row = m;
        self.col = n;
        # U, V should be stored in numpy.matrix form. 
        self.U = np.matrix(np.random.rand(m, r));
        self.V = np.matrix(np.random.rand(r, n));
        
        if self.verbose:
            log('dummy algorithm trained.');
    
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
        
        if not (len(row_idx_arr) == len(col_idx_arr)):
            raise ValueError("The col/row indices of the location should be the same.");
        
        result =  [ (self.U[row, :] * self.V[:, col])[0,0].tolist() for (row, col) in zip(row_idx_arr, col_idx_arr) ];
        if self.verbose:
            log('predicted ' + str(len(row_idx_arr)) + ' elements.');
        
        return result;
    
    
    
        