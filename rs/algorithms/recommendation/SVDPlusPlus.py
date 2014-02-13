'''
Created on Feb 12, 2014

@author: jiayu.zhou
'''

from rs.algorithms.recommendation.generic_recalg import CFAlg
from rs.utils.log import Logger;
import numpy as np;


mcpl_log = lambda message: Logger.Log(SVDPlusPlus.ALG_NAME + ':'+message, Logger.MSG_CATEGORY_ALGO);


class SVDPlusPlus(CFAlg):
    '''
    SVD Plus Plus
    '''
    
    ALG_NAME = 'SVD++';

    def __init__(self, params):
        '''
        Constructor
        '''
        
        self.regularization  = 0.015;
        self.learn_rate      = 0.001;
        self.bais_learn_rate = 0.7;
        self.bais_reg        = 0.33;
        
        self.latent_factors  = 2;
        
        raise NotImplementedError('SVD Plus Plus has not been completed');
        
    def unique_str(self):
        #TODO: complete this.
        return SVDPlusPlus.ALG_NAME + '_k';
    
    def train(self, feedback_data):
        '''
        '''
        self.user_bais = [];
        self.item_bais = [];
        
        
        self.num_user = feedback_data.num_row;
        self.num_item = feedback_data.num_col;
        
        # user factors (related items)
        self.y = [];
        
        # user factors (individual part)
        self.p = [];
        
        self.user_factors = np.zeros((self.num_user, self.latent_factors))
        
        # assign items_rated_by_user = this.ItemRatedByUser();
        
        for user_idx in range(self.user_bais):
            self.PrecomputeFactors(user_idx);
        
        
    def PrecomputedFactors(self, user_id):
        '''
        Pre-compute the factors for a given user. 
        '''
        
        
        
        
        