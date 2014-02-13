'''
Created on Feb 11, 2014

@author: jiayu.zhou
'''

from rs.cache.urm import URM; 
from rs.data.daily_watchtime import DailyWatchTimeReader;
from rs.utils.log import Logger;
from rs.experiments.evaluation import precision_itemlist, recall_itemlist, rmse;
import rs.data.data_split as ds;
import numpy as np;

def experiment_leave_k_out(exp_name, daily_data_file, min_occ_user, min_occ_prog, \
                           method_list, leave_k_out, total_iteration, top_n, binary = False):
    '''
    
    Parameters
    ----------
    @param exp_name: the experiment name (prefix) 
    @param daily_datafile:
    @param min_occ_user:
    
    @param method_list:
    @param leave_k_out: leave k out for each user. The k must be strict less than
         min_occ_user
    
    @param binary: if this is set to true then the binary data is used (non-zero set to 1). 
    
    Returns
    ----------
    @return out 
    '''
    
    if leave_k_out >= min_occ_user:
        raise ValueError('The k in the leave k out should be strictly less than min_occ_user.'); 
    
    # define lko_log style. 
    lko_log = lambda msg: Logger.Log(msg, Logger.MSG_CATEGORY_EXP);
    
    # construct exp_id
    if binary:
        exp_id = 'lko_bi_' + exp_name + '_data' +str(hash(daily_data_file)) + '_mu' + str(min_occ_user) + '_mp' + str(min_occ_prog) \
                      + '_k' + str(leave_k_out) + '_toiter' + str(total_iteration);
    else:
        exp_id = 'lko_' + exp_name + '_data' +str(hash(daily_data_file)) + '_mu' + str(min_occ_user) + '_mp' + str(min_occ_prog) \
                      + '_k' + str(leave_k_out) + '_toiter' + str(total_iteration);
    lko_log('Experiment ID: ' + exp_id);
    
    # load data. 
    lko_log('Read data...');
    reader = DailyWatchTimeReader();
    data = reader.read_file_with_minval(daily_data_file, min_occ_user, min_occ_prog);
    lko_log('Data loaded: ' + str(data));
    
    if binary:
        lko_log('Binarizing data...');
        data.binarize();
    else:
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
            leave_k_out_idx = URM.LoadResource(URM.RTYPE_RESULT, split_resource_str, split_dir);
            if not leave_k_out_idx:
                # randomly generate k items from each row/user.   
                leave_k_out_idx = ds.leave_k_out(data, leave_k_out);
                URM.SaveResource(URM.RTYPE_RESULT, split_resource_str, leave_k_out_idx, split_dir);
            
            # split the k items as a separate. 
            [data_left, data_tr] = data.leave_k_out(leave_k_out_idx); 
            
            iter_result = experiment_unit_leave_k_out(exp_id, method, \
                                    data_tr, data_left, iteration, top_n);
            
            perf_vect.append(iter_result);
    
        result[method.unique_str()] = perf_vect;
    
    return result;
    
def experiment_unit_leave_k_out(exp_id, method, data_tr, data_left, iteration, top_n):
    '''
    This method works on the column/row index of the data_tr and data_left, and 
    the data_tr and data_left must be completely aligned in both row-wise and column-wise. 
    '''
    
    # define lko_log style. 
    lko_log = lambda msg: Logger.Log(msg, Logger.MSG_CATEGORY_EXP);
    
    result_resource_str = 'exp'      + exp_id + \
                          '_method'  + method.unique_str() + \
                          '_iter'    + str(iteration);
    sub_folder = exp_id + '/models/' + method.unique_str(); # use a sub folder to store the experiment resource. 
    
    # check resource for existing model.  
    trained_model = URM.LoadResource(URM.RTYPE_RESULT, result_resource_str, sub_folder);
    if not trained_model:
        
        # train model using the training data. 
        # NOTE: this is the most time-consuming part. 
        lko_log('training models...');
        method.train(data_tr);
        
        # save resource
        trained_model = [method];
        URM.SaveResource(URM.RTYPE_RESULT, result_resource_str, trained_model, sub_folder);
    
    # compute performance on test data using the model.    
    [method] = trained_model;
    lko_log('computing evaluation metrics on the test data...');
    
    eval_result = {};
    # ranked list.
    
    col_num  = data_left.num_col;
    pred_col = range(col_num);
    
    tr_data_csr = data_tr.get_sparse_matrix().tocsr();
    lo_data_csr = data_left.get_sparse_matrix().tocsr();
    
    for user_idx in range(data_left.num_row): 
        # predict the entire row. 
        pred_row = [user_idx] * col_num;
        row_pred = method.predict(pred_row, pred_col);
        # rank the column (the result is a list of indices).
        srt_col = [k[0] for k in sorted(enumerate(row_pred), key=lambda x:x[1], reverse=True)];
        # trained columns.
        tr_col = set(np.nonzero(tr_data_csr[user_idx, :])[1].tolist());
        # remove the trained column.
        te_srt_col = [col_pos for col_pos in srt_col if col_pos not in tr_col];
        # top - k (safeguard)
        te_topk_col = te_srt_col[:min(top_n, len(te_srt_col)-1)]; 
        # test column index;
        lo_col = set(np.nonzero(lo_data_csr[user_idx, :])[1].tolist());
        
        prec = precision_itemlist (te_topk_col, lo_col);
        rec  = recall_itemlist    (te_topk_col, lo_col); 
    eval_result['prec']   = prec;
    eval_result['recall'] = rec;
    eval_result['rmse']   = rmse(data_left.data_val, method.predict(data_left.data_row, data_left.data_col));
    return eval_result;
