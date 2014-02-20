'''
Created on Feb 12, 2014

@author: jiayu.zhou
'''

import numpy as np;
from sklearn import decomposition
from rs.algorithms.recommendation.generic_recalg import CFAlg
from rs.utils.log import Logger; 
from rs.data.daily_watchtime import DailyWatchTimeReader

mcpl_log = lambda message: Logger.Log(NMF.ALG_NAME + ':'+message, Logger.MSG_CATEGORY_ALGO);

class NMF(CFAlg):
    '''
    Recommendation of non-negative factorization. 
    '''
    
    ALG_NAME = 'NMF';

    def __init__(self, latent_factor = 20, beta = 1, eta = 0.1, \
                 maxiter = 100, stop_delta = 0.0001, verbose = False):
        '''
        Constructor
        '''
        self.latent_factor = latent_factor;
        self.beta          = beta;
        self.eta           = eta;
        self.maxiter       = maxiter; 
        self.stop_delta    = stop_delta; 
        self.verbose       = verbose;
        
        self.nmf_solver    = decomposition.ProjectedGradientNMF(n_components=self.latent_factor, \
                            beta = self.beta, eta = self.eta, tol = self.stop_delta, max_iter = self.maxiter, \
                            init='random', random_state = 0 );
        
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
        self.row = feedback_data.num_row;
        self.col = feedback_data.num_col;
        
        data_mat = feedback_data.get_sparse_matrix().tocsr();
        
        print 'Solve H...';
        nmf = self.nmf_solver.fit(data_mat);
        self.H = nmf.components_;
        
        print 'Solve W...';
        self.W = nmf.fit_transform(data_mat);
        
        
        self.W = np.matrix(self.W);
        self.H = np.matrix(self.H);
        
        # process cold start items.
        # 1. find cold start item indices. 
        cs_col = feedback_data.get_cold_start_col();
        Hm = self.H;
        if len(cs_col) > 0:
            # 2. compute column average of V on non-cold start indices.
            ncs_col = list(set(range(self.col)) - set(cs_col));
            Hsum = np.sum(Hm[:, ncs_col], 1)/float(len(ncs_col));
            # 3. fill back to to V.
            Hm[:, cs_col] = Hsum; # assign back to cold start columns.  
        
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
        return (self.W[row_idx, :] * self.H[:, col_idx_arr]).tolist()[0];    
    
if __name__ == '__main__':
    filename = "../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    # load data. 
    reader = DailyWatchTimeReader();  
    feedback_data = reader.read_file_with_minval(filename, 1, 1);
    feedback_data.normalize_row();
    
    # build model with 3 latent factors.
    r = 5;
    
    [f1, f2] = feedback_data.blind_k_out([1]);
    
    NMF_model = NMF(latent_factor = 2);
    NMF_model.train(f1);
        