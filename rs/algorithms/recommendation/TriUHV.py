'''
Created on Jan 31, 2014

@author: Shiyu C. (s.chang1@partner.samsung.com)

'''

import numpy as np;
from rs.algorithms.recommendation.generic_recalg import CFAlg;
from rs.utils.log import Logger; 
import scipy.sparse
import scipy.linalg


# an encapsulated logger.  
log = lambda message: Logger.Log(TriUHV.ALG_NAME + ':'+message, Logger.MSG_CATEGORY_ALGO);




class TriUHV(CFAlg):
    '''
    A random guess recommender (demo).
    '''
    ALG_NAME = 'TriUHV';
    
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
        Train the model with specified feedback_data. 
        WARNING: calling this function will erase previous training information. 
        
        Parameters
        ----------
        feedback_data: feedback data structure in rs.data.recdata.FeedbackData
         
        Returns
        ----------
        no return.
        
        author: Shiyu C. (s.chang1@partner.samsung.com)
        date: Jan 31, 2014
         
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
        # [U,V,S_sparse] = TriUHV.LowRankFitting (S_sparse,U,V,feedback_data,lamb,delta,maxiter);
        
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
    
##################################################################################################
    
    @staticmethod
    def CGF_learnU (S_sparse,U,U_prev,V,V_prev,lamb):
            # Usage:  Cross-Genre Factorization
            # This function fixed S, H and V to learn a proper U
            # author: Shiyu C. (s.chang1@partner,.samsung.com) 
            # date: 01/31/2014
            

            r = U.shape[1];
            I = np.matrix(np.ones((r,r)));
            Inv = np.linalg.inv(V*V.T + lamb*I);
            U_out = (U_prev*(V_prev*V.T))*Inv + S_sparse*(V.T*Inv);
            return U_out;
    
    @staticmethod
    def CGF_learnV (S_sparse,U,U_prev,V,V_prev,lamb):
        # fix the variable S and U and solve V
        r = U.shape[1];
        I = np.matrix(np.ones((r,r)));
        Inv = np.linalg.inv(U.T*U + lamb*I);
        V_out = Inv*(U.T*U_prev)*V_prev + Inv*(U.T*S_sparse);
        return V_out;
    
    @staticmethod
    def CGF_learnS (S_sparse,U,V,feedback_data):
        # learning S with projection 
        val = feedback_data.data_val;
        idx = zip(S_sparse.row.tolist(),S_sparse.col.tolist());
        # idx = zip(feedback_data.data_row,feedback_data.data_col);
        sparse_val = [float(U[i,:]*V[:,j]) for i,j in idx];
        # check if the size of sparse_val equals to the size of data value at the utility matrix
        if len(sparse_val) != len(val):
            print "The length of computed value is not matching the number of non-zero entry in X."
            raise ValueError()    
        temp = np.matrix(val) - np.matrix(sparse_val);
        temp2 = temp.tolist();
        temp3 = temp2[0];
        temp3 = np.array(temp3);
        S_sparse.data = temp3;
        return S_sparse;
    
    
    @staticmethod
    def LowRankFitting (S_sparse,U,V,X,lamb,delta,maxiter):
        # the objective function is given as
        # min \|S - UV\|_F^2, s.t. \mathcal{P}(S) = \mathcal{P}(X)
        # input: the utility matrix X
        # output: S,U and V
        counter = 0;
        U_prev = U;
        V_prev = V;
        while counter < maxiter :
            # print counter
            counter += 1;
            print "Iteration: ", counter;
            U = TriUHV.CGF_learnU(S_sparse,U,U_prev,V,V_prev,lamb);    
            print "update U"
            temp = (U_prev*V_prev + S_sparse) - U*V_prev;
            obj = scipy.linalg.norm(temp)**2;
            print "objective: ", obj;
            
            print "Update V"
            V = TriUHV.CGF_learnV(S_sparse,U,U_prev,V,V_prev,lamb); 
            temp = (U_prev*V_prev + S_sparse) - U*V;
            obj = scipy.linalg.norm(temp)**2;
            print "objective: ", obj;
            
            print "Update S"
            S_sparse = TriUHV.CGF_learnS(S_sparse,U,V,X);
    
            # calculate the objective function 
            obj = scipy.linalg.norm(np.matrix(S_sparse.data),2)**2;
            print "The objective value: ", obj;
            # print scipy.linalg.norm(S_sparse.todense())**2;
            
            # check the termination condition
            checker = max(abs(U-U_prev).max(),abs(V-V_prev).max());
            print "The delta value: ", checker
            # print max(abs(U-U_prev).max(),abs(V-V_prev).max());
            if checker <= delta:
                break;
            U_prev = U;
            V_prev = V;
            
        return [U,V,S_sparse];