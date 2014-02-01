'''
This is a testing pipeline for KDD_2014 algorithm.
 
Created on Jan 30, 2014

@author: Shiyu C. (s.chang1@partner.samsung.com)
'''

from rs.data.daily_watchtime import DailyWatchTimeReader
from rs.algorithms.recommendation.TriUHV import TriUHV

if __name__ == '__main__':
    filename = "../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    # load data. 
    reader = DailyWatchTimeReader();  
    feedback_data = reader.read_file_with_minval(filename, 1, 1);
    
    # build model with 3 latent factors.
    r = 10;
    # the L_2 norm regularizer 
    lamb = 0.001; 
    # the stopping delta value 
    delta = 1;
    # the maximium iteration number
    maxiter = 1;
     
    TriUHV_model = TriUHV(r,lamb,delta,maxiter, verbose = True); 
    TriUHV_model.train(feedback_data);
'''    
    # test. 
    loc_row = [200,   4, 105];
    loc_col = [ 10,  22,   4];
    print 'Prediction:'
    print TriUHV_model.predict(loc_row, loc_col);
'''    
    