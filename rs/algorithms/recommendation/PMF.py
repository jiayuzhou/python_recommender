'''
Created on Jan 31, 2014

@author: Shiyu C. (s.chang1@partner.samsung.com)

'''

import numpy as np;
from rs.algorithms.recommendation.generic_recalg import CFAlg;
from rs.utils.log import Logger; 
#import scipy.sparse;
#import scipy.linalg;
from rs.algorithms.recommendation.ProbabilisticMatrixFactorization import ProbabilisticMatrixFactorization;


# an encapsulated logger.  
log = lambda message: Logger.Log(PMF.ALG_NAME + ':'+message, Logger.MSG_CATEGORY_ALGO);




class PMF(CFAlg):
    '''
    The probabilistic matrix factorization (PMF) algorithm.  
    '''
    ALG_NAME = 'PMF';
    
##################################################################################################

    def __init__(self, latent_factor = 20, lamb = 1e-3, learning_rate = 0.0001, maxiter = 100, verbose = False):
        '''
        Constructor
        '''
        # initialize parameters. 
        self.latent_factor = latent_factor;
        self.lamb = lamb;
        self.maxiter = maxiter;
        self.learn_rate = learning_rate;
        
        log('PMF algorithm instance created: latent factor ' + str(self.latent_factor));
         
        self.verbose = verbose;
    
    def unique_str(self):
        return PMF.ALG_NAME + '_k' + str(self.latent_factor) + '_lamb' + str(self.lamb)  \
            + '_maxiter' + str(self.maxiter) + '_lr' + str(self.learn_rate);
    
##################################################################################################
        
    def train(self, feedback_data):
        '''
        Training stage, use/modify the online source at file ProbabilisticMatrixFactorization 
        
        '''
        if self.verbose:
            log('training PMF algorithm.');
        
        m = feedback_data.num_row;
        n = feedback_data.num_col;  
        r = self.latent_factor;
        maxiter = self.maxiter;
        
        self.row = m;
        self.col = n;
        
        # U, V should be stored in numpy.matrix form. 
        # initialization of U, V and S_sparse
        
        # U = np.matrix(np.random.rand(m, r));
        # V = np.matrix(np.random.rand(r,n));   
        
        feedback_data.normalize_row();     
        # S_sparse = scipy.sparse.coo_matrix((np.array(feedback_data.data_val,dtype = np.float64),(feedback_data.data_row,feedback_data.data_col)),(m,n));    
        
        ratings =  zip(feedback_data.data_row,feedback_data.data_col, feedback_data.data_val);
        
        pmf = ProbabilisticMatrixFactorization(ratings, latent_d=r, \
                                lamb = self.lamb, learning_rate = self.learn_rate);
                                
        liks = []
        
        counter = 0;
        while (pmf.update() and counter<maxiter):
            counter += 1;
            print 'Iteration: ', counter;
            lik = pmf.likelihood()
            liks.append(lik)
            print "L=", lik
            pass
        
    
        self.U = np.matrix(pmf.users);
        self.V = np.matrix((pmf.items).T);

        
        if self.verbose:
            log('PMF algorithm trained.');
            
##################################################################################################
    
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
        return (self.U[row_idx, :] * self.V[:, col_idx_arr]).tolist()[0];    
    
    