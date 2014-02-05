'''
This module consists of recommender wrappers for factorization methods. 

Created on Feb 4, 2014

@author: jiayu.zhou
'''


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

    