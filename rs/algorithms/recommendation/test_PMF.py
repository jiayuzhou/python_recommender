'''
This is a testing pipeline for PMF algorithm.
 
Created on Feb 11, 2014

@author: Shiyu C. (s.chang1@partner.samsung.com)
'''

from rs.data.daily_watchtime import DailyWatchTimeReader
from rs.algorithms.recommendation.PMF import PMF

if __name__ == '__main__':
    filename = "../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    # load data. 
    reader = DailyWatchTimeReader();
    feedback_data = reader.read_file_with_minval(filename, 1, 1);
    
    # build model with 3 latent factors.
    r = 10;
    # the L_2 norm regularizer 
    lamb = 0.1; 
    # the stopping delta value 
    delta = 1e-5;
    # the maximium iteration number
    maxiter = 300;
     
    PMF_model = PMF(r,lamb,delta,maxiter, verbose = True);
    print PMF_model.unique_str();
    
    PMF_model.train(feedback_data);
    
    # test. 
    loc_row = [200,   4, 105];
    loc_col = [ 10,  22,   4];
    print 'Prediction:'
    print PMF_model.predict(loc_row, loc_col);
    
    