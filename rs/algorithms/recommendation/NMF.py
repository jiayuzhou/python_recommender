'''
Created on Feb 12, 2014

@author: jiayu.zhou
'''

import numpy as np;
from sklearn import decomposition
from rs.algorithms.recommendation.generic_recalg import CFAlg
from rs.utils.log import Logger; 

mcpl_log = lambda message: Logger.Log(NMF.ALG_NAME + ':'+message, Logger.MSG_CATEGORY_ALGO);

class NMF(CFAlg):
    '''
    Recommendation of non-negative factorization. 
    '''
    
    ALG_NAME = 'NMF';

    def __init__(self, latent_factor = 20, beta = 1, eta = 0.1, \
                 maxiter = 200, stop_delta = 0.0001, verbose = False):
        '''
        Constructor
        '''
        self.latent_factor = latent_factor;
        self.beta          = beta;
        self.eta           = eta;
        self.maxiter       = maxiter; 
        self.stop_delta    = stop_delta; 
        self.verbose       = verbose;
        
        self.nmf_solver    = decomposition.NMF(n_components=self.latent_factor, \
                            beta = self.beta, eta = self.eta, tol = self.stop_delta, max_iter = self.maxiter );
        
        mcpl_log('NMF instance created: latent factor ' + str(self.latent_factor));
        
    def unique_str(self):
        return NMF.ALG_NAME + '_k' + str(self.latent_factor) + '_beta' + str(self.beta) + '_eta' + str(self.eta)\
            + '_maxIter' + str(self.maxiter) + '_delta' + str(self.stop_delta);
    
    def train(self, feedback_data):
        '''
        Parameters
        ----------
        @param feedback_data: feedback data structure
        
        Returns
        ----------
        no return.  
        '''
        
        data_mat = feedback_data.get_sparse_matrix().tocsr();
        
        print 'Solve H...';
        nmf = self.nmf_solver.fit(data_mat);
        self.H = nmf.components_;
        
        print 'Solve W...';
        self.W = nmf.fit_transform(data_mat);
        
        self.H = np.matrix(self.H);
        self.W = np.matrix(self.W);
        
        if self.verbose:
            mcpl_log('NMF model trained.');
        
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
        
        result =  [ (self.W[row, :] * self.H[:, col])[0,0].tolist() for (row, col) in zip(row_idx_arr, col_idx_arr) ];
        if self.verbose:
            mcpl_log('predicted ' + str(len(row_idx_arr)) + ' elements.');
        
        return result;
        