'''

Specified training and testing with mean average precision 

Created on Apr 29, 2014

@author: jiayu.zhou
'''

from rs.cache.urm import URM; 
from rs.data.utility_data import UtilityDataReader;
from rs.utils.log import Logger;
from rs.experiments.evaluation import rmse;
import numpy as np;

def experiment_tr_te_map(exp_name, train_data_file,  test_data_file, \
                        train_item_feature_file, test_item_feature_file, \
                        max_rank, binary = False):
    '''
    Parameters 
    ----------
    @param exp_name:           
    @param train_data_file:    
    @param test_data_file:     
    @param train_content_file: 
    @param test_content_file:  
    @param max_rank:           the maximal N in the computation. 
    
    Returns
    ----------
    @return out
    '''
    
    # initialize utilities 
    trte_log = lambda msg: Logger.Log(msg, Logger.MSG_CATEGORY_EXP)
    
    # processing file name hashing (used for cache string).
    #   create hash for single file or a list of files. 
    if isinstance(train_data_file, list):    
        hash_file_tr_data_str = str(hash(tuple(train_data_file)));
    else:
        hash_file_tr_data_str = str(hash(train_data_file));   
    
    if isinstance(test_data_file, list):    
        hash_file_te_data_str = str(hash(tuple(test_data_file)));
    else:
        hash_file_te_data_str = str(hash(test_data_file));
    
    if train_item_feature_file:
        if isinstance(train_item_feature_file, list):
            hash_file_tr_item_feature_str = str(hash(tuple(train_item_feature_file)))
        else:
            hash_file_tr_item_feature_str = str(hash(train_item_feature_file))
    else:
        hash_file_tr_item_feature_str = '';
        
    if test_item_feature_file:
        if isinstance(test_item_feature_file, list):
            hash_file_te_item_feature_str = str(hash(tuple(test_item_feature_file)))
        else:
            hash_file_te_item_feature_str = str(hash(test_item_feature_file))
    else:
        hash_file_te_item_feature_str = '';    
    
    # display information 
    print 'Training data file', train_data_file, ' [hash:', hash_file_tr_data_str, ']'
    if train_item_feature_file:
        print 'Training content feature provided: ', train_item_feature_file, \
                 ' [hash:', hash_file_tr_item_feature_str, ']'
    else:
        print 'Training content feature not provided.'
         
    print 'Testing data file ', test_data_file, ' [hash:', hash_file_te_data_str, ']'
    if test_item_feature_file:
        print 'Testing content feature provided: ', test_item_feature_file, \
                 ' [hash:', hash_file_te_item_feature_str, ']'
    else:
        print 'Testing content feature not provided.'
    
    if binary:
        exp_id_prefix = 'trte_bi_'
    else:
        exp_id_prefix = 'trte_'
    
    exp_id = exp_id_prefix + exp_name + '_trdata_' + hash_file_tr_data_str \
                                      + '_tedata_' + hash_file_te_data_str \
                                      + '_tritemf_' + hash_file_tr_item_feature_str \
                                      + '_tritemf_' + hash_file_te_item_feature_str;
    trte_log('Experiment ID: ' + exp_id)
    
    # load utility data and feature data. 
    trte_log('Read training data...')
    reader = UtilityDataReader(fieldDelimiter = '\t');
    
    tr_data = reader.read_file_with_minval(train_data_file, 0, 0);
    trte_log('Training data loaded: '+ str(tr_data))
    
    te_data = reader.read_file_with_minval(test_data_file, 0, 0);
    trte_log('Testing data loaded: '+ str(te_data))
    
    # load item feature data 
    
    
    if binary:
        trte_log('Binarizing data...');
        tr_data.binarize();
        te_data.binarize();
    
    result = {};
    