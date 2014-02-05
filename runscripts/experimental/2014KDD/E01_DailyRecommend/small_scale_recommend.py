'''
Created on Feb 4, 2014

@author: jiayu.zhou
'''

from rs.algorithms.recommendation.LMaFit import Rec_LMaFit;
from rs.experiments.dwt_rec_futurehit import experiment_future_program;
if __name__ == '__main__':
    daily_data_file1 = "../../../../datasample/agg_duid_pid_watchtime_genre/toy_small_day1";
    daily_data_file2 = "../../../../datasample/agg_duid_pid_watchtime_genre/toy_small_day1";
    #daily_data_file2 = "../../../../datasample/agg_duid_pid_watchtime_genre/toy_small_day2";
    
    exp_name = 'test_small_rec' ;
    
    # filtering 
    min_occ_user = 0;
    min_occ_prog = 0;
    
    rec_alg1 = Rec_LMaFit(latent_factors = [4], lamb = [1e-5]);
    
    # methods. 
    method_list = [ rec_alg1 ]; 
    
    result = experiment_future_program(exp_name, daily_data_file1, daily_data_file2,\
                                        min_occ_user, min_occ_prog, method_list, 2);
                                        
    print result;