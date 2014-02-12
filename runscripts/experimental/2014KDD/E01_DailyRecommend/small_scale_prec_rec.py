'''
Created on Feb 3, 2014

@author: jiayu.zhou
'''

from rs.experiments.dwt_rec_leave_N_out import experiment_leave_k_out;
from rs.algorithms.recommendation.LMaFit import LMaFit;
from rs.algorithms.recommendation.RandUV import RandUV;
from rs.algorithms.recommendation.HierLat import HierLat


if __name__ == '__main__':
    daily_data_file = "../../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    exp_name = 'test_exp_mid_pr'; # something meaningful. 
    
    # filtering criteria
    min_occ_user = 35;
    min_occ_prog = 300;
    
    top_n = 10; # performance computed on top N; 
    
    leave_k_out = 10; # perform leave k out. 
    
    # number of repetitions. 
    total_iteration = 3;
    
    # latent factor
    lf = 5;
    
    # recommendation algorithms 
    method_list = [ LMaFit(latent_factor = 5), RandUV(latent_factor = 5), HierLat(latent_factor = 5)  ];
    
    # main method. 
    result = experiment_leave_k_out(exp_name, daily_data_file, min_occ_user, min_occ_prog, \
                method_list,  leave_k_out, total_iteration, top_n);
    
    # display results (average RMSE). 
    for method_name, method_iter_perf in result.items():
        print 'Method: '+ method_name;
        print  '>>Average precision : %.5f' % (sum( x['prec']   for x in method_iter_perf)/len(method_iter_perf));
        print  '>>Average recall    : %.5f' % (sum( x['recall'] for x in method_iter_perf)/len(method_iter_perf));
        print  '>>Average rmse      : %.5f' % (sum( x['rmse']   for x in method_iter_perf)/len(method_iter_perf));
        #print method_iter_perf;
    
    #print result;