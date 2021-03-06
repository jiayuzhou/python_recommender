'''
Experiment computing 

watchtime setting. 

Created on Feb 15, 2014

@author: jiayu.zhou
'''

import sys;
import numpy as np;
import scipy.io as sio;
from rs.algorithms.recommendation.LMaFit  import LMaFit;
from rs.algorithms.recommendation.HierLat import HierLat;
from rs.algorithms.recommendation.NMF     import NMF;
from rs.algorithms.recommendation.PMF     import PMF
from rs.algorithms.recommendation.TriUHV  import TriUHV;
from rs.experiments.dwt_rec_leave_N_out_map import experiment_coldstart_map;

if __name__ == '__main__':
    
    #daily_data_file = "../../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    # latent factor
    lafactor = 5;
    leave_k_out = 1; # perform leave k out.

#     daily_data_file = [
#                     '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140201/part-r-00000',
#                 ]
#     exp_name = 'exp_map_weekly';
#     min_occ_user = 50;
#     min_occ_prog = 500;
    
    daily_data_file = ['/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140201/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140202/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140203/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140204/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140205/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140206/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140207/part-r-00000'
                       ]
    exp_name = 'exp_map_Feb1to7_coldstart_time';
    # filtering criteria.
    min_occ_user = 50;
    min_occ_prog = 1000;
    
    num_user = 10000;
    num_prog = 3000;
    
    if not len(sys.argv) == 1:
        leave_k_out = int(sys.argv[1]);
    print 'Use default leave k out: k=' + str(leave_k_out);
    
    if not len(sys.argv) <= 2:
        lafactor = int(sys.argv[2]);
    print 'Use default latent factor: ' + str(lafactor); 
    
    max_rank = 2000;
    
    # number of repetitions. 
    total_iteration = 2;
    
    # recommendation algorithms RandUV(latent_factor=lafactor),
    method_list = [ HierLat(latent_factor=lafactor, cold_start = HierLat.CS_EQUAL_PROB),  \
                    LMaFit(latent_factor=lafactor),  NMF(latent_factor=lafactor),
                    PMF(latent_factor=lafactor),     TriUHV(latent_factor=lafactor)  ];
    
    # main method. 
    result = experiment_coldstart_map(exp_name, daily_data_file, \
                min_occ_user, min_occ_prog, num_user, num_prog,\
                method_list,  leave_k_out, total_iteration, max_rank, binary = False);
    
    matlab_output = {};
    for method_name, method_iter_perf in result.items():
        print 'Method: '+ method_name;
        rmse = sum( x['RMSE']   for x in method_iter_perf)/len(method_iter_perf);
        print  '>>Average RMSE      : %.5f' % rmse;
        
        perf_recall    = np.zeros(len(method_iter_perf[0]['recall']));
        perf_precision = np.zeros(len(method_iter_perf[0]['precision'])); 
        
        for x in method_iter_perf:
            perf_recall    += np.array(x['recall']);
            perf_precision += np.array(x['precision']);
            
        perf_precision = perf_precision / len(method_iter_perf);
        perf_recall    = perf_recall    / len(method_iter_perf); 
        
        # convert to valid MATLAB name. 
        matlab_var_name = method_name.replace('.', '_').replace(' ', '_');
        matlab_output['REC_'  + matlab_var_name] = perf_recall;
        matlab_output['PREC_' + matlab_var_name] = perf_precision;
        matlab_output['RMSE_' + matlab_var_name] = rmse; 
        
    # save to file.
    hash_file_str = str(hash(tuple(daily_data_file))); 
    matlab_file = 'cst_bi_' + exp_name + '_data' + hash_file_str + '_mu' + str(min_occ_user) + '_mp' + str(min_occ_prog) \
                      + '_k' + str(leave_k_out) + '_toiter' + str(total_iteration) + '_fa' + str(lafactor);
    sio.savemat(matlab_file, matlab_output);
    