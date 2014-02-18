'''
Created on Feb 17, 2014

@author: Shiyu C. (s.chang1@partner.samsung.com)

'''

import numpy as np;
from rs.algorithms.recommendation.generic_recalg import CFAlg;
from rs.utils.log import Logger; 
import scipy.sparse;
#import scipy.linalg

# an encapsulated logger.  
log = lambda message: Logger.Log(item_item_sim.ALG_NAME + ':'+message, \
                                 Logger.MSG_CATEGORY_ALGO);

class item_item_sim(CFAlg):
    
    ALG_NAME = 'item_item_sim';
    

    def __init__(self, N = 3):
        '''
        Constructor
        '''
        # initialize parameters. 
        self.N = N;
        log('Item-based similarity algorithm created: Neighborhood size' + str(self.N));
    
        
    def unique_str(self):
        return item_item_sim.ALG_NAME + '_neighborhood' + str(self.N);
        
        
    def train(self, feedback_data):
        '''
        Sim will be a np.matrix type;
        '''        
        N = self.N;

        num_user = feedback_data.num_row;
        num_item = feedback_data.num_col;
        S = scipy.sparse.csr_matrix((np.array(feedback_data.data_val,dtype = np.float64),\
                                     (feedback_data.data_row,feedback_data.data_col)),(num_user,num_item));
        affinity = (S.T*S).todense();
        
        
        # initialization of the output S;                             
        S_out = S.todense();
        
        for i in range(num_user):    
        # for i in [0]:
                
            print 'Iteration: ', i, 'out of ', num_user
            # obtain the list of non_zero item entry of a certain user        
            nz_col = S.indices[S.indptr[i]:S.indptr[i+1]];
            
            # take out the non_zero value of S
            # Val = np.squeeze(S[i,nz_col].todense().tolist());
            Val = np.array(S[i,nz_col].todense())[0].tolist();
            Val = np.array(Val);
            # print 'Value', Val           
            
            
            if len(nz_col) == 0:
                # means all column value is equal to zero
                # this case should not happen!!!!!
                #################################################
                # NOTE: currently not handling, otherwise, should
                # be all random value
                #################################################
                continue;        
            # print nz_col
    
            # check the top value whether is valid or not        
            if len(nz_col) < N:
                top = len(nz_col);
            else:
                top = N;
                
            z_col = set(range(S.shape[1])).difference(set(nz_col));
            z_col = list(z_col);
            
            if len(z_col) == 0:
                # means all column has value already no need to fill 
                # the missing entry for the current user
                continue;
            
            if len(nz_col) == 1:
                # This means for a particular user, there is only on column with value
                # we have to use the value to fill out the rest of entries
                ####################################################
                # NOTES: This implementation is correct, however, the algorithm itself
                # does not make too much sense.
                # Either try to avoid user with only 1 ratings or have to think anothor
                # way to CF the missing entries
                ###################################################
                # check if 'Val' is a single value
                if len(Val) != 1: 
                    raise ValueError("something is wrong.");
        
                S_out[i,z_col] = Val;
                continue;
            
            
                
            # need to fill out the rest of entries with proper value
            for j in range(len(z_col)):
                
                
                # print 'inner iteration: ', j
                # t corresponding to the real idx in the S_matrix
                t = z_col[j];
                
                # take out the cosine-similarity of them 
                sim = affinity[t,nz_col];
                
                # print sim
                # sort these similarity value;
                sort_sim = np.array(np.sort(sim)).reshape(-1,).tolist()[::-1][0:top];
                sim_idx = np.array(np.argsort(sim)).reshape(-1,).tolist()[::-1][0:top];
                # print sim;
                # print sort_sim
                # print sim_idx
                sort_sim = np.array(sort_sim);
                sim_idx = np.array(sim_idx);
                
                # compute the CF value            
                # S_out[i,t] = sum(np.multiply(sort_sim / float(sum(sort_sim)), sort_sim));
                if (sum(sort_sim) == 0):
                    S_out[i,t] = 0;
                    continue;
                
                S_out[i,t] = sum(np.multiply(sort_sim / float(sum(sort_sim)), Val[sim_idx]));
                # S_out[i,t] = -1.32;
                # print S.todense();
                # print S_out;   
                     
        print 'Training completed'
        self.S_out = S_out;


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
        
        result =  [ self.S_out[row,col] for (row, col) in zip(row_idx_arr, col_idx_arr) ];
        
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
        
        result =  [ self.S_out[row,col] for (row, col) in zip([row_idx] * len(col_idx_arr), col_idx_arr) ];
        return result;     
        
   