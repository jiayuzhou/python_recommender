'''
This is a testing pipeline for KDD_2014 algorithm.
 
Created on Jan 30, 2014

@author: Shiyu C. (s.chang1@partner.samsung.com)
'''

from rs.data.daily_watchtime import DailyWatchTimeReader
from rs.algorithms.recommendation.TriUHV import TriUHV
from rs.data.recdata import FeedbackData

if __name__ == '__main__':
    filename = "../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    # load data. 
    reader = DailyWatchTimeReader();  
    feedback_data = reader.read_file_with_minval(filename, 1, 1);
    feedback_data.normalize_row();
    
    # build model with 3 latent factors.
    r = 5;
    # the L_2 norm regularizer 
    lamb = 0.2; 
    # the stopping delta value 
    delta = 0.01;
    # the maximium iteration number
    maxiter = 500;
     
#     TriUHV_model = TriUHV(r,lamb,delta,maxiter, verbose = True); 
#     TriUHV_model.train(feedback_data);
    
    '''    
    # test. 
    loc_row = [200,   4, 105];
    loc_col = [ 10,  22,   4];
    print 'Prediction:'
    print TriUHV_model.predict(loc_row, loc_col);
    '''
        
    # test cold start.
    row =  [0, 0, 1, 1, 2, 3, 3];
    col =  [0, 3, 1, 4, 0, 1, 3];
    data = [1, 1, 1, 1, 1, 1, 1];
     
    [f1, f2] = feedback_data.blind_k_out([1]);
    
    LMaFit_model = TriUHV(r,lamb,delta,5, verbose = True);
    LMaFit_model.train(f1);
    
    
    
    