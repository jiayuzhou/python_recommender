'''
Created on Feb 3, 2014

@author: jiayu.zhou
'''

from rs.experiments.util_rec import experiment_rand_split;
from rs.algorithms.recommendation.LMaFit  import LMaFit;
from rs.algorithms.recommendation.RandUV  import RandUV;
from rs.algorithms.recommendation.NMF     import NMF;
from rs.algorithms.recommendation.PMF     import PMF
from rs.algorithms.recommendation.item_item_sim import item_item_sim


if __name__ == '__main__':
    daily_data_file = "../../../datasample/thomas_data/user_to_item_rating_sample.tsv";
    
    exp_name = 'loc_rec_exp'; # something meaningful. 
    
    # filtering criteria
    min_occ_user = 4;
    min_occ_prog = 1;
    
    # specify the percentage of training and (1 - training_prec) is testing.
    training_prec = 0.5;
    
    # number of repetitions. 
    total_iteration = 3;
    
    lafactor = 5;
    
    # recommendation algorithms 
    method_list = [ LMaFit(latent_factor=lafactor),  item_item_sim(N = lafactor), \
                    NMF(latent_factor=lafactor), \
                    PMF(latent_factor=lafactor),  \
                    RandUV(latent_factor=lafactor)  \
                ];
    
    # main method. 
    result = experiment_rand_split(exp_name, daily_data_file, min_occ_user, min_occ_prog, \
                method_list,  training_prec, total_iteration);
    
    # display results (average RMSE). 
    for method_name, method_iter_perf in result.items():
        print 'Method: '+ method_name;
        print  '>>Average performance RMSE: %.5f' % (sum( x for x in method_iter_perf)/len(method_iter_perf));
    
    #print result;