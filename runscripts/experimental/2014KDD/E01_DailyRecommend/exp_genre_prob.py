'''
Explore explicit genre space.  

Created on Feb 19, 2014

@author: jiayu.zhou
'''
from rs.algorithms.recommendation.HierLat import HierLat
from rs.cache.urm import URM;
from rs.data.daily_watchtime import DailyWatchTimeReader
import scipy.io as sio;

if __name__ == '__main__':
    ### Generate resource location. 
    
    daily_data_file = ['/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140201/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140202/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140203/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140204/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140205/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140206/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140207/part-r-00000'
                       ]
    exp_name = 'exp_map_Feb1to7_randsplit_bin';
    
    # filtering criteria.
    min_occ_user = 50;
    min_occ_prog = 1000;
    
    num_user = 10000;
    num_prog = 3000;
    
    total_iteration = 2;
    
    iteration = 1; # iteration out of total_iteration. 
    
    leave_k_out = 20;
    lafactor = 5;
    
    method = HierLat(latent_factor=lafactor);
    hash_file_str = str(hash(tuple(daily_data_file)));
    
    reader = DailyWatchTimeReader();
    feedback_data = reader.read_file_with_minval(daily_data_file, min_occ_user, min_occ_prog, num_user, num_prog);
        
    exp_id = 'lko_bi_' + exp_name + '_data' + hash_file_str\
                      + '_mu' + str(min_occ_user) + '_mp' + str(min_occ_prog) \
                      + '_nu' + str(num_user) + '_np' + str(num_prog) \
                      + '_k' + str(leave_k_out) + '_toiter' + str(total_iteration);
    
    result_resource_str = 'exp'      + exp_id + \
                          '_method'  + method.unique_str() + \
                          '_iter'    + str(iteration);
    sub_folder = exp_id + '/models/' + method.unique_str(); # use a sub folder to store the experiment resource.
                          
    trained_model = URM.LoadResource(URM.RTYPE_RESULT, result_resource_str, sub_folder);
    [method] = trained_model;
    
    learnt_genre = method.V;
    
    program_mapping = feedback_data.col_mapping;
    program_inv_mapping = {y: x for x, y in program_mapping.items()};
    program_name = [ program_inv_mapping[i] for i in range(len(program_mapping)) ];
    
    sio.savemat("prog_genre_mat.mat", {'genre_mat': learnt_genre, 'prog_name': program_name});
    
    print 'done';
    
    
    
    