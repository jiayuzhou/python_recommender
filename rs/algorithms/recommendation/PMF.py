'''
Created on Jan 31, 2014

@author: Shiyu C. (s.chang1@partner.samsung.com)

'''

import numpy as np;
from rs.algorithms.recommendation.generic_recalg import CFAlg;
from rs.utils.log import Logger; 
import scipy.sparse;
import scipy.linalg



# an encapsulated logger.  
log = lambda message: Logger.Log(PMF.ALG_NAME + ':'+message, Logger.MSG_CATEGORY_ALGO);




class PMF(CFAlg):
    '''
    A random guess recommender (demo).
    '''
    ALG_NAME = 'PMF';
    
##################################################################################################

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
        
##################################################################################################
        
    def train(self, feedback_data):
        '''
        Training stage, use/modify the online source at file ProbabilisticMatrixFactorization 
        
        '''
        if self.verbose:
            log('training dummy algorithm.');
        
        # load the meta data including the genre information to properly store them 
        meta = feedback_data.meta;
#        row_mapping = feedback_data.row_mapping;
#        col_mapping = feedback_data.col_mapping;
        
        m = feedback_data.num_row;
        n = feedback_data.num_col;  
        r = self.latent_factor;
        g = 330;
        #print meta
        #print type(meta['pggr_pg']);
        #print type(meta['pggr_gr']);
        if len(meta['pggr_pg']) != len(meta['pggr_gr']):
            raise ValueError("The length of the meta data mismatched.")
        V_val = [1 for i in range(len(meta['pggr_pg']))];
        # print V_val 
            
        lamb = self.lamb;
        delta = self.delta;
        maxiter = self.maxiter;
                
        self.row = m;
        self.col = n;
        
        # U, H, V should be stored in numpy.matrix form. 
        # initialization of U, H, V and S_sparse
        U = np.matrix(np.random.rand(m, r));
        H = np.matrix(np.random.rand(r,g));
        V = scipy.sparse.coo_matrix((V_val,(meta['pggr_gr'],meta['pggr_pg'])),shape = (g,n));
        # print V.shape
        # print V     
        S_sparse = scipy.sparse.coo_matrix((np.array(feedback_data.data_val,dtype = np.float64),(feedback_data.data_row,feedback_data.data_col)),(m,n));
        # S_sparse = S_sparse.tolil();
        # print S_sparse.row

        ###############################
        # the main learning process
        # U = PMF.CGF_learnU (S_sparse,U,U,H,H,V,V,lamb);
        # H = PMF.CGF_learnH (S_sparse,U,U,H,H,V,V);
        # V = PMF.CGF_learnV (S_sparse,U,U,H,H,V,V,lamb);
        # S_sparse = PMF.CGF_learnS (S_sparse,U,H,V,feedback_data);
        
        [U,H,V,S_sparse] = PMF.Learn (S_sparse,U,H,V,feedback_data,lamb,delta,maxiter);
        
        self.U = U;
        self.H = H;
        self.V = V;
        self.S_sparse = S_sparse;
        
        if self.verbose:
            log('dummy algorithm trained.');
            
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
    