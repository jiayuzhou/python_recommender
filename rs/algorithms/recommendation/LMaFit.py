'''
Created on Jan 24, 2014

@author: Shiyu C. (s.chang1@partner.samsung.com)
'''

import numpy as np;
from rs.algorithms.recommendation.generic_recalg import CFAlg;
from rs.utils.log import Logger; 
import scipy.sparse


# an encapsulated logger.  
log = lambda message: Logger.Log(LMaFit.ALG_NAME + ':'+message, Logger.MSG_CATEGORY_ALGO);




class LMaFit(CFAlg):
    '''
    A random guess recommender (demo).
    '''
    ALG_NAME = 'LMaFit';

    def __init__(self, latent_factor = 20, lamb = 1e-3, stop_delta = 1e-4, maxiter = 1e3, verbose = False):
        '''
        Constructor
        '''
        # initialize parameters. 
        self.latent_factor = latent_factor;
        self.lamb = lamb;
        self.delta = stop_delta; 
        self.maxiter = maxiter;
        
        log('dummy algorithm instance created: latent factor ' + str(self.latent_factor));
        
        self.verbose = verbose;
        
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
        lamb = self.lamb;
        
        self.row = m;
        self.col = n;
        
        # U, V should be stored in numpy.matrix form. 
        # initialization of U, V and S_sparse
        U = np.matrix(np.random.rand(m, r));
        V = np.matrix(np.random.rand(r,n));
        S_sparse = scipy.sparse.coo_matrix((np.array(feedback_data.data_val),(np.array(feedback_data.data_row), np.array(feedback_data.data_col))),(m,n));
        S1 = S_sparse.tocsr();
        S2 = S_sparse * S_sparse.transpose();
        print S_sparse;
        U = LMaFit.LRF_learnU (S_sparse,U,V,lamb);
        
        
        
        
        
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
    
    @staticmethod
    def LRF_learnU (S_sparse,U,V,lamb):
            # fix the variable S and V solve U
            r = U.shape[1];
            I = np.matrix(np.ones((r,r)));
            Inv = np.linalg.pinv(V*V.T + lamb*I);
            U_out = (U*(V*V.T))*Inv + S_sparse*(V.T*Inv);
            return U_out1 + U_out2;
    
    