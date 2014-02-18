'''
Experiment computing 

Binary Setting.

Created on Feb 15, 2014

@author: jiayu.zhou
'''

import sys;
import numpy as np;
import scipy.io as sio;
from rs.algorithms.recommendation.LMaFit  import LMaFit;
from rs.algorithms.recommendation.RandUV  import RandUV;
from rs.algorithms.recommendation.HierLat import HierLat;
from rs.algorithms.recommendation.NMF     import NMF;
from rs.algorithms.recommendation.PMF     import PMF
from rs.algorithms.recommendation.TriUHV  import TriUHV;
from rs.experiments.dwt_rec_leave_N_out_map import experiment_leave_k_out_map

if __name__ == '__main__':
    
    #daily_data_file = "../../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    # latent factor
    lafactor = 5;
    leave_k_out = 1; # perform leave k out.

#     daily_data_file = [
#                     '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140201/part-r-00000',
#                 ]
#     exp_name = 'exp_map_weekly_bin';
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
    exp_name = 'exp_map_Feb1to7_randsplit_bin';
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
    
    # recommendation algorithms 
    method_list = [ LMaFit(latent_factor=lafactor), RandUV(latent_factor=lafactor), \
                    HierLat(latent_factor=lafactor), NMF(latent_factor=lafactor),
                    PMF(latent_factor=lafactor),     TriUHV(latent_factor=lafactor)  ];
    
    # main method. 
    result = experiment_leave_k_out_map(exp_name, daily_data_file, \
                min_occ_user, min_occ_prog, num_user, num_prog,\
                method_list,  leave_k_out, total_iteration, max_rank, binary = True);
    
    matlab_output = {};
    for method_name, method_iter_perf in result.items():
        print 'Method: '+ method_name;
        print  '>>Average rmse      : %.5f' % (sum( x['rmse']   for x in method_iter_perf)/len(method_iter_perf));
        
        perf = np.zeros(len(method_iter_perf[0]['map']));
        for x in method_iter_perf:
            perf += np.array(x['map']);
            
        perf = perf / len(method_iter_perf); 
        
        #print '>>map:   ' 
        #print perf.tolist();
        
        matlab_output[method_name.replace('.', '_')]=perf; 
            
    # save to file.
    hash_file_str = str(hash(tuple(daily_data_file))); 
    matlab_file = 'lko_bi_' + exp_name + '_data' + hash_file_str + '_mu' + str(min_occ_user) + '_mp' + str(min_occ_prog) \
                      + '_k' + str(leave_k_out) + '_toiter' + str(total_iteration);
    sio.savemat(matlab_file, matlab_output);