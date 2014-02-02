'''
Recommendation performance evaluation. 

Created on Feb 2, 2014

@author: jiayu.zhou
'''

from rs.cache.urm import URM; # load resource manager. 

def experiment_rand_split(method_list, method_str_list, daily_data_file, training_prec, total_iteration):
    
    exp_id = 'some experiment id that associated ';
    # save experiment splitting as resources. 
    
    for method_idx, method in enumerate(method_list):
        
        # do for each method
        for iteration in range(total_iteration):
            # do for each iteration for each method;
            iter_result = experiment_unit_rand_split(method, method_str_list[method_idx], \
                            daily_data_file, training_prec, iteration);        
    

def experiment_unit_rand_split(exp_id, method, method_str, tr_data, te_data, iteration):
    '''
    
    '''
    result_resource_str = exp_id + hash(method) + iteration;
    
    # check resource.  
    if URM.CheckResource(URM.RTYPE_RESULT, result_resource_str):
        return URM.LoadResource(URM.RTYPE_RESULT, result_resource_str);
    
    model  = [];
    perf   = [];
    
    result = [perf, model];
    
    # save resource
    URM.SaveResource(URM.RTYPE_RESULT, result_resource_str, result);
    
    
    
    
    
    