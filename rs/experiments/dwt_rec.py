'''
Recommendation performance evaluation using daily watch time (dwt) data.  

The experimental setting is to perform On-Demand TV program recommendation. 
Random splitting a feedback data, and use the training part to train
and testing part to test the algorithm. 

Created on Feb 2, 2014

@author: jiayu.zhou
'''

from rs.cache.urm import URM; # load resource manager. 
import rs.data.data_split as ds;
from rs.data.daily_watchtime import DailyWatchTimeReader
from rs.experiments.evaluation import rmse;
from rs.utils.log import Logger;

def experiment_rand_split(exp_name, daily_data_file, min_occ_user, min_occ_prog, \
                          method_list,  training_prec, total_iteration):
    '''
    
    '''
    # define log style. 
    log = lambda msg: Logger.Log(msg, Logger.MSG_CATEGORY_EXP);
    
    #log('Data ID: ' + hash(daily_data_file));
    
    # here we use a regular hash. 
    exp_id = exp_name + '_data' +str(hash(daily_data_file)) + '_mu' + str(min_occ_user) + '_mp' + str(min_occ_prog) \
                      + '_trprec' + str(training_prec) + '_toiter' + str(total_iteration);
    
    log('Experiment ID: ' + exp_id);
    
    # save experiment splitting as resources. 
    reader = DailyWatchTimeReader();
    data = reader.read_file_with_minval(daily_data_file, min_occ_user, min_occ_prog);
    
    # we normalize here before splitting.
    log('Normalizing data...'); 
    data.normalize_row();
    
    result = {};
    
    for method in method_list:
    # do for each method
        
        perf_vect = [];
        for iteration in range(total_iteration):
        # do for each iteration for each method;
            
            log('Method: '+ method.unique_str() + ' Iteration: '+ str(iteration));
            
            # data split of the current iteration. 
            split_resource_str = 'exp' + exp_id + '_split_iter' + str(iteration); 
            split_dir = exp_id + '/split';
            split = URM.LoadResource(URM.RTYPE_RESULT, split_resource_str, split_dir);
            if not split:
                split = ds.split(data.num_row, training_prec);
                URM.SaveResource(URM.RTYPE_RESULT, split_resource_str, split, split_dir);
            
            [split_tr, split_te] = split;
            data_tr = data.subdata(split_tr);
            data_te = data.subdata(split_te);
            
            iter_result = experiment_unit_rand_split(exp_id, method, data_tr, data_te, iteration);
                            
            perf_vect.append(iter_result);
       
        result[method.unique_str()] = perf_vect;
        
    log('Experiment Done [' + exp_id + ']');
    
    return result;

def experiment_unit_rand_split(exp_id, method, tr_data, te_data, iteration):
    '''
    One iteration of training and testing. The experimental ID 
    '''
    
    # define log style. 
    log = lambda msg: Logger.Log(msg, Logger.MSG_CATEGORY_EXP);
    
    result_resource_str = 'exp'      + exp_id + \
                          '_method'  + method.unique_str() + \
                          '_iter'    + str(iteration);
    sub_folder = exp_id + '/models/' + method.unique_str(); # use a sub folder to store the experiment resource. 
    
    # check resource for existing model.  
    trained_model = URM.LoadResource(URM.RTYPE_RESULT, result_resource_str, sub_folder);
    if not trained_model:
        
        # train model using the training data. 
        # NOTE: this is the most time-consuming part. 
        log('training models...');
        method.train(tr_data);
        
        # save resource
        trained_model = [method];
        URM.SaveResource(URM.RTYPE_RESULT, result_resource_str, trained_model, sub_folder);
    
    # compute performance on test data using the model.    
    [method] = trained_model;
    log('computing evaluation metrics on the test data...');
    eval_result = rmse(te_data.data_val, method.predict(te_data.data_row, te_data.data_col));
    
    return eval_result;
    
    
    
    