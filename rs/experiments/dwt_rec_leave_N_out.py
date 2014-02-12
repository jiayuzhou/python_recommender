'''
Created on Feb 11, 2014

@author: jiayu.zhou
'''

from rs.cache.urm import URM; 
from rs.data.daily_watchtime import DailyWatchTimeReader;
from rs.utils.log import Logger;
import rs.data.data_split as ds;

def experiment_leave_k_out(exp_name, daily_data_file, min_occ_user, min_occ_prog, \
                           method_list, leave_k_out, total_iteration):
    '''
    
    Parameters
    ----------
    @param exp_name: the experiment name (prefix) 
    @param daily_datafile:
    @param min_occ_user:
    
    @param method_list:
    @param leave_k_out: leave k out for each user. The k must be strict less than
         min_occ_user
    
    
    Returns
    ----------
    @return out 
    '''
    
    if leave_k_out >= min_occ_user:
        raise ValueError('The k in the leave k out should be strictly less than min_occ_user.'); 
    
    # define lko_log style. 
    lko_log = lambda msg: Logger.Log(msg, Logger.MSG_CATEGORY_EXP);
    
    # construct exp_id
    exp_id = 'lko_' + exp_name + '_data' +str(hash(daily_data_file)) + '_mu' + str(min_occ_user) + '_mp' + str(min_occ_prog) \
                      + '_k' + str(leave_k_out) + '_toiter' + str(total_iteration);
    lko_log('Experiment ID: ' + exp_id);
    
    # load data. 
    lko_log('Read data...');
    reader = DailyWatchTimeReader();
    data = reader.read_file_with_minval(daily_data_file, min_occ_user, min_occ_prog);
    lko_log('Data loaded: ' + str(data));
    
    # normalize 
    lko_log('Normalizing data...');
    data.normalize_row();
    
    result = {};
    
    for method in method_list:
        # do for each method
    
        perf_vect = [];
        for iteration in range(total_iteration):
            # do for each iteration for each method. 
    
            lko_log('Method: '+ method.unique_str() + ' Iteration: '+ str(iteration));
    
            # data split of the current iteration. 
            split_resource_str = 'exp' + exp_id + '_lvidx_iter' + str(iteration); 
            split_dir = exp_id + '/lv_idx';
            split = URM.LoadResource(URM.RTYPE_RESULT, split_resource_str, split_dir);
            if not split:
                # TODO: check the correctness. 
                leave_k_out_idx = ds.leave_k_out(data, leave_k_out);
                URM.SaveResource(URM.RTYPE_RESULT, split_resource_str, split, split_dir);
                
            [data_left, data_tr] = data.leave_k_out(leave_k_out_idx); 
            
            iter_result = experiment_unit_leave_k_out(exp_id, method, data_tr, data_left, iteration);
            
            perf_vect.append(iter_result);
    
    result[method.unique_str()] = perf_vect;
    
    
    
    
    