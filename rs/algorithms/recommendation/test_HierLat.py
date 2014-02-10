'''
This is a testing pipeline for KDD_2014 algorithm.
 
Created on Jan 30, 2014

@author: Shiyu C. (s.chang1@partner.samsung.com)
'''
import numpy as np;
from rs.data.daily_watchtime import DailyWatchTimeReader
from rs.algorithms.recommendation.HierLat import HierLat

if __name__ == '__main__':
    
    
    # load data. 
    reader = DailyWatchTimeReader();
    
    #filename = "../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";  
    #feedback_data = reader.read_file_with_minval(filename, 1, 1);
    
    
    filename = "/Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131209/part-r-00000";
    feedback_data = reader.read_file_with_minval(filename, 25, 300);
    print feedback_data;
    
    print 'Maximum Genre.'
    print np.max(feedback_data.meta['pggr_gr']) + 1;
    
    print 'Normalizing data.'
    feedback_data.normalize_row();
    
    # build model with 3 latent factors.
    r = 5;
    # the L_2 norm regularizer 
    lamb = 0.2; 
    # the stopping delta value 
    delta = 0.01;
    # the maximum iteration number
    maxiter = 500;
     
    HierLat_model = HierLat(r,lamb,delta,maxiter, verbose = True); 
    #HierLat_model.train(feedback_data, simplex_projection = False);
    HierLat_model.train(feedback_data, simplex_projection = True);
'''    
    # test. 
    loc_row = [200,   4, 105];
    loc_col = [ 10,  22,   4];
    print 'Prediction:'
    print HierLat_model.predict(loc_row, loc_col);
'''    
    