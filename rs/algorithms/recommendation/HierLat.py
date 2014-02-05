'''
Created on Feb 5, 2014

@author: Shiyu C. (s.chang1@partner.samsung.com)

'''

import numpy as np;
from rs.algorithms.recommendation.generic_recalg import CFAlg;
from rs.utils.log import Logger; 
import scipy.sparse;
import scipy.linalg



# an encapsulated logger.  
log = lambda message: Logger.Log(HierLat.ALG_NAME + ':'+message, Logger.MSG_CATEGORY_ALGO);




class HierLat(CFAlg):
    '''
    A random guess recommender (demo).
    '''
    ALG_NAME = 'HierLat';
    
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
        
#         ###############################
#         # check the gradient 
#         S_U = np.matrix(np.random.rand(m, r));
#         S_H = np.matrix(np.random.rand(r,g));
#         S_V = scipy.sparse.coo_matrix((V_val,(meta['pggr_gr'],meta['pggr_pg'])),shape = (g,n));
#         HierLat.check_Vgradient (S_sparse,U,S_U,H,S_H,V,S_V,lamb)         

        ###############################
        # the main learning process
        # U = HierLat.CGF_learnU (S_sparse,U,U,H,H,V,V,lamb);
        # H = HierLat.CGF_learnH (S_sparse,U,U,H,H,V,V);
        # V = HierLat.CGF_learnV (S_sparse,U,U,H,H,V,V,lamb);
        # S_sparse = HierLat.CGF_learnS (S_sparse,U,H,V,feedback_data);
        
        [U,H,V,S_sparse] = HierLat.Learn (S_sparse,U,H,V,feedback_data,lamb,delta,maxiter);
     
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
    def df_V (V,prod1,prod2,prod3,lamb): 
        '''
        This function the gradient direction for the current value of U,H,V and S_sparse 
        also we need the S_U, S_H and S_V which is the UHV value associated with S since 
        S can be decomposed as a low-rank matrix plus a sparse matrix 
        For an efficient implementation, given the product of H^T*U^T*S_U*S_H*S_V 
        ,(UH)^T*S_sparse and (UH)^T*UH    
        --------------------------
        Inputs
        V: the current V value
        prod1: H^T*U^T*S_U*S_H*S_V 
        prod2: (UH)^T*S_sparse
        prod3: (UH)^T*UH   
        
        Returns 
        
        The gradient:
        -2(UH)^T(S_U*S_H*S_V + S_sparse - UHV) + 2lambV
        
        --------------------------
        Copyright: Shiyu C. (s.chang1@partner.samsung.com)        
        '''
        
        df = -2*prod1 - 2*prod2 + 2*prod3*V + 2*lamb*V;
         
        return df;
    
##################################################################################################
    @staticmethod 
    def Objval_V (Objprod,Objprod4,Objprod5,Objprod6,V,lamb):    
        '''
        This function calculate the objective function of \| S - UHV \|_2^2 + \lambda \|V \\_2^2
        
        --------------------------
        Inputs
        V: the current V value
        
        Objprod: Objprod1 + Objprod2 + Objprod3  where Objprod1,Objprod2 and Objprod3 is listed below 
        
        Objprod1 = np.trace((S_sparse.T * S_sparse).todense());
        Objprod2 = 2*np.trace(S_sparse.T*S_U*S_H*S_V);
        Objprod3 = np.trace(S_V.T*S_H.T*(S_U.T*S_U)*(S_H*S_V));
        
        and
        
        Objprod4 = S_sparse.T*U*H;
        Objprod5 = S_V.T*S_H.T*(S_U.T*U)*H;
        Objprod6 = H.T*U.T*U*H;
           
        lamb: the regularization parameter
        
        --------------------------
        Outputs
            
        f -- The objective function of \| S - UHV \|_2^2 + \lambda \|V\|_2^2  
         
        Copyright: Shiyu C. (s.chang1@partner.samsung.com)
         
        '''

#         Objprod4 = S_sparse.T*U*H;
#         Objprod5 = S_V.T*S_H.T*(S_U.T*U)*H;
#         Objprod6 = H.T*U.T*U*H;
        
        f = Objprod - 2*np.trace(Objprod4*V) - 2*np.trace(Objprod5*V) + np.trace(V.T*Objprod6*V);
        f += lamb*scipy.linalg.norm(V.todense())**2;
        
        return f;
      
##################################################################################################
    @staticmethod 
    def Objval_V_danteng (S_sparse,U,S_U,H,S_H,V,S_V,lamb):    
        '''
        This function calculate the objective function of \| S - UHV \|_2^2 + \lambda \|V \\_2^2 
        '''
        temp = (S_U*S_H*S_V + S_sparse) - U*H*V;
        f = scipy.linalg.norm(temp)**2;
        f += lamb*scipy.linalg.norm(V.todense())**2;
        return f;
    
##################################################################################################
    @staticmethod
    def check_Vgradient (S_sparse,U,S_U,H,S_H,V,S_V,lamb): 
        
        [a,b] = V.todense().shape;
        delta = np.matrix(np.random.rand(a,b));
        delta = delta / np.linalg.norm(delta);
        
        Objprod1 = np.trace((S_sparse.T * S_sparse).todense());
        Objprod2 = 2*np.trace(S_sparse.T*S_U*S_H*S_V);
        Objprod3 = np.trace(S_V.T*S_H.T*(S_U.T*S_U)*(S_H*S_V));
        Objprod = Objprod1 + Objprod2 + Objprod3; 
        
        Objprod4 = S_sparse.T*U*H;
        Objprod5 = S_V.T*S_H.T*(S_U.T*U)*H;
        Objprod6 = H.T*U.T*U*H;
           
           
        f0 = HierLat.Objval_V (Objprod,Objprod4,Objprod5,Objprod6,V,lamb);
        
        df_prod1 = H.T*U.T*S_U*S_H*S_V; 
        df_prod2 = (U*H).T*S_sparse;
        df_prod3 =  (U*H).T*U*H;    
        
        df0 = HierLat.df_V(V,df_prod1,df_prod2,df_prod3,lamb);
        
        epsilon = [0.1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7];
        
        for i in range(len(epsilon)):
            V_left = scipy.sparse.coo_matrix((V - epsilon[i]*delta));
            V_right = scipy.sparse.coo_matrix(V + epsilon[i]*delta);
            f_left = HierLat.Objval_V (Objprod,Objprod4,Objprod5,Objprod6,V_left,lamb);
            f_right = HierLat.Objval_V (Objprod,Objprod4,Objprod5,Objprod6,V_right,lamb);
            Vs = (f_right - f_left) / 2;
            Vs_hat = df0.flatten(1) * ((epsilon[i]*delta).flatten(1)).T;
            print epsilon[i], Vs/Vs_hat;

        return ;
    
##################################################################################################
    
    @staticmethod
    def learnU (S_sparse,U,U_prev,H,H_prev,V,V_prev,lamb):
        # Usage:  Cross-Genre Factorization
        # This function fixed S, H and V to learn a proper U
        # author: Shiyu C. (s.chang1@partner.samsung.com) 
        # date: 01/31/2014
        
        r = U.shape[1];
        I = np.matrix(np.eye(r));
        # temp = H*V*V.T*H.T;
        Inv = np.linalg.inv(H*V*V.T*H.T + lamb*I);         
        HV_transpose = V.T*H.T;
        U_out = (U_prev*H_prev)*(V_prev*HV_transpose)*Inv + S_sparse*HV_transpose*Inv;
        return U_out;
        
##################################################################################################
        
    @staticmethod
    def learnH (S_sparse,U,U_prev,H,H_prev,V,V_prev):
        # fix the variable S and U, V and solve for H
        r = U.shape[1];
        g = V.shape[0];
        Inv1 = np.linalg.pinv(U.T*U); # An rxr matrix
        Inv2 = np.linalg.pinv((V*V.T).todense());  # An gxg matrix
        H_out = Inv1*(U.T*U_prev)*H_prev*(V_prev*V.T)*Inv2 + Inv1*((U.T*S_sparse)*V.T)*Inv2;
        return H_out;
    
################################################################################################## 
   
    @staticmethod
    def learnV (S_sparse,U,U_prev,H,H_prev,V,V_prev,lamb):
        # fix the variable S and U H and solve V
        ######################
        # Calculate the closed form solution
        ######################
        g = V.shape[0];
        I = np.matrix(np.eye(g));
        # print H.T*(U.T*U)*H + lamb*I;
        # print H.T*(U.T*U)*H + lamb*I;
        Inv = np.linalg.inv(H.T*(U.T*U)*H + lamb*I);
        V_out = (Inv*H.T)*(U.T*U_prev)*(H_prev*V_prev) + Inv*H.T*(U.T*S_sparse);
        V_out = scipy.sparse.coo_matrix(V_out); 
        ######################
        # Projection (eliminate the non-zero entries)
        ######################
#         NZrow = V.row;
#         NZcol = V.col;
#         # print NZrow;
#         # print NZcol;
#         Val = [V_out[i,j] for (i,j) in zip(NZrow, NZcol)];
#         # print Val;
#         V_out = scipy.sparse.coo_matrix((Val,(NZrow,NZcol)),shape = (g,V.shape[1]));
#         # print V_out;
        return V_out;
    
##################################################################################################    
    @staticmethod
    def learnS (S_sparse,U,H,V,feedback_data):
        # learning S with projection 
        val = feedback_data.data_val;
        idx = zip(S_sparse.row.tolist(),S_sparse.col.tolist());
        # idx = zip(feedback_data.data_row,feedback_data.data_col);
        HV = H*V;
        sparse_val = [float(U[i,:]*HV[:,j]) for i,j in idx];
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
    
##################################################################################################    
    
    @staticmethod
    def Learn (S_sparse,U,H,V,X,lamb,delta,maxiter):
        # the objective function is given as
        # min \|S - UV\|_F^2, s.t. \mathcal{P}(S) = \mathcal{P}(X)
        # input: the utility matrix X
        # output: S,U,H and V
        counter = 0;
        U_prev = U;
        H_prev = H;
        V_prev = V;
        while counter < maxiter :
            # print counter
            counter += 1;
            print "Iteration: ", counter;
            print "Update U"
            U = HierLat.learnU(S_sparse,U,U_prev,H,H_prev,V,V_prev,lamb);    
#             temp = (U_prev*H_prev*V_prev + S_sparse) - U*H_prev*V_prev;
#             obj = scipy.linalg.norm(temp)**2;
#             obj += lamb*(scipy.linalg.norm(U)**2 + scipy.linalg.norm(V_prev.todense())**2);
#             print "objective: ", obj;
            
            print "Update H"
            H = HierLat.learnH(S_sparse,U,U_prev,H,H_prev,V,V_prev);    
#             temp = (U_prev*H_prev*V_prev + S_sparse) - U*H*V_prev;
#             obj = scipy.linalg.norm(temp)**2;
#             obj += lamb*(scipy.linalg.norm(U)**2 + scipy.linalg.norm(V_prev.todense())**2);
#             print "objective: ", obj;
            
            
            print "Update V"
            V = HierLat.learnV(S_sparse,U,U_prev,H,H_prev,V,V_prev,lamb); 
#             temp = (U_prev*H_prev*V_prev + S_sparse) - U*H*V;
#             obj = scipy.linalg.norm(temp)**2;
#             obj += lamb*(scipy.linalg.norm(U)**2 + scipy.linalg.norm(V.todense())**2);
#             print "objective: ", obj;
            
            print "Update S"
            S_sparse = HierLat.learnS(S_sparse,U,H,V,X);
    
            # calculate the objective function 
            obj1 = scipy.linalg.norm(np.matrix(S_sparse.data),2)**2;
            obj2 = lamb*(scipy.linalg.norm(U)**2 + scipy.linalg.norm(V.todense())**2)
            obj = obj1 + obj2;
            print "The objective value: ", obj;
            # print scipy.linalg.norm(S_sparse.todense())**2;
            
            # check the termination condition
            checker = max(abs(U-U_prev).max(),abs(V-V_prev).max());
            print "The delta value: ", checker
            # print max(abs(U-U_prev).max(),abs(V-V_prev).max());
            if checker <= delta:
                print "Termination condition meet"
                break;
            U_prev = U;
            H_prev = H;
            V_prev = V;
            
        return [U,H,V,S_sparse];
    
    