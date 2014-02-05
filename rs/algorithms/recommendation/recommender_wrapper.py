'''
This module consists of recommender wrappers for factorization methods. 

Created on Feb 4, 2014

@author: jiayu.zhou
'''

import numpy as np;

class Recommender(object):
    '''
    The interface for recommender wrappers. One recommender may maintain 
    one or a set of machine learning models (e.g., a latent model, or an ensemble 
    of latent models/memory-based models). 
    '''

    def train(self, feedback_data):
        '''
        Train model/models using the specified feedback_data. 
        '''
        raise NotImplementedError("Interface method.");

    def get_score(self, user_id, item_id_list, meta_data = None):
        '''
        Get the scores of the items whose IDs are in the item_id_list. The 
        meta_data contains additional information of the items/users.  
        '''
        raise NotImplementedError("Interface method.");

    def unique_str(self):
        '''
        Output the unique string to identify the type and configuration of this recommender.    
        '''
        raise NotImplementedError("Interface method.");


class Rec_Dummy(Recommender):
    '''
    An example of recommender.
    
    This is a dummy recommender which returns whatever seen in the data set or return 0. 
    '''
    
    def __init__(self):
        '''
        really nothing to do. 
        '''
        pass;
    
    def train(self, feedback_data):
        '''
        record the Feedback data set. 
        '''
        self.data         = feedback_data.get_sparse_matrix().tolil();
        self.user_mapping = feedback_data.row_mapping;
        self.item_mapping = feedback_data.col_mapping;
        
    def get_score(self, user_id, item_id_list, meta_data = None):
        '''
        user_id: string.
        '''
        if not user_id in self.user_mapping:
            print 'Cold start user: ' + user_id;
            return np.zeros(len(item_id_list)).tolist();
        
        score = [self.data[self.user_mapping[user_id], self.item_mapping[item_id]] \
                        if item_id in self.item_mapping else 0 for item_id in item_id_list ];
        
        return score;
        
    def unique_str(self):
        return 'Rec_Dummy';
        
        