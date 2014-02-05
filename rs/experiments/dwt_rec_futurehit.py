'''
This evaluates the prediction on future programs (where the program ID might not have seen before).

The experimental setting is to perform recommendation on future (regular) TV programs. 
The previous data is used to construct a recommendation model, and the future data (the next day) 
data is used to evaluate the recommendation.    

Created on Feb 4, 2014

@author: jiayu.zhou
'''

from rs.cache.urm import URM; # load resource manager. 
from rs.data.daily_watchtime import DailyWatchTimeReader;
from rs.utils.log import Logger;
from rs.experiments.evaluation import hit_prec;
#import numpy as np; 

def experiment_future_program(exp_name, previous_data_files, future_data_file, \
                              min_occ_user, min_occ_prog, method_list, top_k):
    '''
    experiment entrance for future programs.
    
    Top-k precision. 
    
    Parameters
    ----------
    exp_name:    a human-readable experiment name.
    method_list: a list of recommendation models  
    
    Returns
    ---------- 
    '''
    # define mcpl_log style. 
    mcpl_log = lambda msg: Logger.Log(msg, Logger.MSG_CATEGORY_EXP);
    
    exp_id = exp_name + '_mu' + str(min_occ_user) + '_mp' + str(min_occ_prog);
    
    mcpl_log('Experimental ID: ' + exp_id);
    
    reader = DailyWatchTimeReader();
    tr_data = reader.read_file_with_minval(previous_data_files, min_occ_user, min_occ_prog);
    te_data = reader.read_file_with_minval(future_data_file,    min_occ_user, min_occ_prog);
    
    mcpl_log('Normalization data ...');
    tr_data.normalize_row();
    # there is no need to normalize train data because we evaluate the hits. 
    
    result = {}; 
    
    for method in method_list:
    # do for each method 
    
        mcpl_log('Method: '+ method.unique_str());
        method_result = experiment_unit_future_program(exp_id, method, tr_data, te_data, top_k);
        
        result[method.unique_str()] = method_result; 
        
    mcpl_log('Experiment Done [' + exp_id + ']');
    
    return result;
        
        
def experiment_unit_future_program(exp_id, method, tr_data, te_data, top_k):
    '''
    '''
    
    # define mcpl_log style. 
    mcpl_log = lambda msg: Logger.Log(msg, Logger.MSG_CATEGORY_EXP);
    
    result_resource_str = 'model_exp'      + exp_id + \
                          '_method'  + method.unique_str();
    sub_folder = exp_id + '/models/' + method.unique_str(); # use a sub folder to store the experiment resource. 
    
    # check resource for existing model.  
    trained_model = URM.LoadResource(URM.RTYPE_RESULT, result_resource_str, sub_folder);
    if not trained_model:
        
        # train model using the training data. 
        # NOTE: this is the most time-consuming part. 
        mcpl_log('training models...');
        method.train(tr_data);
        
        # save resource
        trained_model = [method];
        URM.SaveResource(URM.RTYPE_RESULT, result_resource_str, trained_model, sub_folder);
    
    # compute performance on test data using the model.    
    [method] = trained_model;
    mcpl_log('computing evaluation metrics on the test data...');
    
    # compute the score of the programs in the prediction.
    prog_list  = te_data.col_mapping.keys(); # program list 
    
    te_datamat = te_data.get_sparse_matrix().tolil();
    
    eval_result = [];
    
    # TODO: on a subset of DUID? 
    for duid in te_data.row_mapping.keys(): # iterate every element. 
        prog_score = method.get_score(duid, prog_list, te_data.meta);  # get scores of the programs in the list. 
        
        # sort the score (first dimension is the index and the second is the actual prediction value).
        #    NOTE: the first dimension is the order with respect to prog_list
        srt_list = [(k[0], k[1]) for k in sorted(enumerate(prog_score), key=lambda x:x[1], reverse=True)];
        
        srt_list = srt_list[:top_k]; # truncate to top k. 
        
        [srt_idx, _]  = zip(*srt_list);
        
        # map from prog_list to actual index. 
        mapped_srt_idx = [te_data.col_mapping[prog_list[idx]] for idx in srt_idx];
        
        #print te_datamat[te_data.row_mapping[duid], mapped_srt_idx].todense();
        
        # get the ground truth hit.
        prog_hit = (te_datamat[te_data.row_mapping[duid], mapped_srt_idx].todense().tolist())[0];
        
        # compute hit precision (now we consider only binary hit).  
        eval_result.append(hit_prec(prog_hit));
    
    return eval_result; # the result is an array containing the precision of every user. 




