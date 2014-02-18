'''
Leave-N-out cross-validation with mean average precision.  

Created on Feb 11, 2014

@author: jiayu.zhou
'''

from rs.cache.urm import URM; 
from rs.data.daily_watchtime import DailyWatchTimeReader;
from rs.utils.log import Logger;
from rs.experiments.evaluation import rmse;
import rs.data.data_split as ds;
import numpy as np;

def experiment_leave_k_out_map(exp_name,     daily_data_file,\
                    min_occ_user, min_occ_prog, num_user, num_prog,\
                    method_list, leave_k_out, total_iteration, max_rank, binary = False):
    '''
    
    Parameters
    ----------
    @param exp_name:       the experiment name (prefix) 
    @param daily_datafile: a list of files. 
    @param min_occ_user:   cold start user criteria
    @param min_occ_prog:   cold start user criteria
    @param num_user:       the number of users selected in the experiment. 
    @param num_prog:       the number of programs selected in the experiment. 
    @param method_list:
    @param leave_k_out: leave k out for each user. The k must be strict less than
         min_occ_user
    
    @param binary: if this is set to true then the binary data is used (non-zero set to 1). 
    
    Returns
    ----------
    @return out 
    '''
    
    print 'Leave k out: k = ', str(leave_k_out);
    print 'Min_occ_user: ',    str(min_occ_user);
    print 'Min_occ_prog: ',    str(min_occ_prog);
    
    if leave_k_out >= min_occ_user:
        raise ValueError('The k in the leave k out [' + str(leave_k_out) 
                         +'] should be strictly less than min_occ_user [' + str(min_occ_user) +'].'); 
    
    # define lko_log style. 
    lko_log = lambda msg: Logger.Log(msg, Logger.MSG_CATEGORY_EXP);
    
    
    if isinstance(daily_data_file, list):    
        hash_file_str = str(hash(tuple(daily_data_file)));
    else:
        hash_file_str = str(hash(daily_data_file));
    
    # construct exp_id
    if binary:
        exp_id = 'lko_bi_' + exp_name + '_data' + hash_file_str\
                      + '_mu' + str(min_occ_user) + '_mp' + str(min_occ_prog) \
                      + '_nu' + str(num_user) + '_np' + str(num_prog) \
                      + '_k' + str(leave_k_out) + '_toiter' + str(total_iteration);
    else:
        exp_id = 'lko_'    + exp_name + '_data' + hash_file_str\
                      + '_mu' + str(min_occ_user) + '_mp' + str(min_occ_prog) \
                      + '_nu' + str(num_user) + '_np' + str(num_prog) \
                      + '_k' + str(leave_k_out) + '_toiter' + str(total_iteration);
    lko_log('Experiment ID: ' + exp_id);
    
    # load data. 
    lko_log('Read data...');
    reader = DailyWatchTimeReader();
    data = reader.read_file_with_minval(daily_data_file, min_occ_user, min_occ_prog, num_user, num_prog);
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
            
            iter_result = experiment_unit_leave_k_out_map(exp_id, method, \
                                    data_tr, data_left, iteration, max_rank);
            
            perf_vect.append(iter_result);
    
        result[method.unique_str()] = perf_vect;
    
    return result;
    
def experiment_unit_leave_k_out_map(exp_id, method, data_tr, data_left, iteration, max_rank):
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
    
    perf_vect_prec = np.zeros(max_rank); # precision 
    perf_vect_rec  = np.zeros(max_rank); # recall 
    perf_vect_hr   = np.zeros(max_rank); # hit rate (Modification of Xia Ning's Paper) 
    
    for user_idx in range(data_left.num_row): 
        # predict the entire row. 
        
        #pred_row = [user_idx] * col_num;
        #row_pred = method.predict(pred_row, pred_col);
        row_pred = method.predict_row(user_idx, pred_col);
        
        # rank the column (the result is a list of indices).
        srt_col = [k[0] for k in sorted(enumerate(row_pred), key=lambda x:x[1], reverse=True)];
        
        # trained columns.
        tr_col = set(np.nonzero(tr_data_csr[user_idx, :])[1].tolist());
        
        # test column index;
        lo_col = set(np.nonzero(lo_data_csr[user_idx, :])[1].tolist());
        
        # remove the trained column from prediction. 
        # this contains a set of indices that predicted (excluding training items).
        te_srt_col = [col_pos for col_pos in srt_col if col_pos not in tr_col];
        
        #max_rank will result in an array of 0:max_rank-1;
        
        hit = 0; # the hit variable keeps track of the number of hits till the current rank. 
        
        for rk in range(max_rank):
            # if rk is greater than the length of te_srt_col, then continue;
            # if not, detect possible hits.
            #    a hit is defined by items hits  
            if (rk < len(te_srt_col)) and (te_srt_col[rk] in lo_col):
                hit += 1;
            
            perf_vect_hr[rk]   += float(hit)/len(lo_col); # hit rate
            perf_vect_prec[rk] += float(hit)/(rk+1);          # precision
            perf_vect_rec[rk]  += float(hit)/len(lo_col); # recall

    #normalization over users.
    perf_vect_hr   = perf_vect_hr/data_left.num_row; 
    perf_vect_prec = perf_vect_prec/data_left.num_row;
    perf_vect_rec  = perf_vect_rec/data_left.num_row;
         
    eval_result['hit_rate']  = perf_vect_hr;
    eval_result['precision'] = perf_vect_prec; 
    eval_result['recall']    = perf_vect_rec; 
    eval_result['RMSE']      = rmse(data_left.data_val, method.predict(data_left.data_row, data_left.data_col));
    return eval_result;
