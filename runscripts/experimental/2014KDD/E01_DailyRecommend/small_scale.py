'''
Created on Feb 3, 2014

@author: jiayu.zhou
'''

from rs.experiments.dwt_rec import experiment_rand_split;
from rs.algorithms.recommendation.LMaFit import LMaFit;
from rs.algorithms.recommendation.RandUV import RandUV;
from rs.algorithms.recommendation.HierLat import HierLat


if __name__ == '__main__':
    daily_data_file = "../../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    exp_name = 'test_exp'; # something meaningful. 
    
    # filtering criteria
    min_occ_user = 4;
    min_occ_prog = 1;
    
    # specify the percentage of training and (1 - training_prec) is testing.
    training_prec = 0.5;
    
    # number of repetitions. 
    total_iteration = 3;
    
    # recommendation algorithms 
    method_list = [ LMaFit(latent_factor = 2), RandUV(latent_factor = 2), HierLat(latent_factor = 2)  ];
    
    # main method. 
    result = experiment_rand_split(exp_name, daily_data_file, min_occ_user, min_occ_prog, \
                method_list,  training_prec, total_iteration);
    
    # display results (average RMSE). 
    for method_name, method_iter_perf in result.items():
        print 'Method: '+ method_name;
        print  '>>Average performance RMSE: %.5f' % (sum( x for x in method_iter_perf)/len(method_iter_perf));
    
    #print result;