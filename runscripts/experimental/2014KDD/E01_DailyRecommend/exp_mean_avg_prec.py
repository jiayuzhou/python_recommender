'''
Experiment computing 

Created on Feb 15, 2014

@author: jiayu.zhou
'''

import numpy as np;
from rs.algorithms.recommendation.LMaFit  import LMaFit;
from rs.algorithms.recommendation.RandUV  import RandUV;
from rs.algorithms.recommendation.HierLat import HierLat;
from rs.algorithms.recommendation.NMF     import NMF;
from rs.experiments.dwt_rec_leave_N_out_map import experiment_leave_k_out_map

if __name__ == '__main__':
    
    #daily_data_file = "../../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    daily_data_file = ['/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140201/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140202/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140203/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140204/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140205/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140206/part-r-00000',
                       '/hadoop05/home/jiayu.zhou/data/agg_duid_pid_watchtime_genre/20140207/part-r-00000'
                       ]
    
    exp_name = 'exp_map_weekly_bin';
    
    # filtering criteria. 
    min_occ_user = 50;
    min_occ_prog = 300;
    
    max_rank = 1000;
    
    leave_k_out = 1; # perform leave k out.
    
    # number of repetitions. 
    total_iteration = 2;
    
    # latent factor
    lf = 5;
    
    # recommendation algorithms 
    method_list = [ LMaFit(latent_factor = 5), RandUV(latent_factor = 5), \
                   HierLat(latent_factor = 5) , NMF(latent_factor = 5) ];
    
    # main method. 
    result = experiment_leave_k_out_map(exp_name, daily_data_file, min_occ_user, min_occ_prog, \
                method_list,  leave_k_out, total_iteration, max_rank, binary = True);
    
    for method_name, method_iter_perf in result.items():
        print 'Method: '+ method_name;
        print  '>>Average rmse      : %.5f' % (sum( x['rmse']   for x in method_iter_perf)/len(method_iter_perf));
        
        perf = np.zeros(len(method_iter_perf[0]['map']));
        for x in method_iter_perf:
            perf += np.array(x['map']);
            
        perf = perf / len(method_iter_perf); 
        
        print '>>map:   ' 
        print perf.tolist();
            
    