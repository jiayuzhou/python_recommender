'''
This is a testing pipeline for KDD_2014 algorithm.
 
Created on Feb 17, 2014

@author: Shiyu C. (s.chang1@partner.samsung.com)
'''

from rs.data.daily_watchtime import DailyWatchTimeReader
from rs.algorithms.recommendation.item_item_sim import item_item_sim

if __name__ == '__main__':
    filename = "../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    # load data. 
    reader = DailyWatchTimeReader();  
    feedback_data = reader.read_file_with_minval(filename, 1, 1);
    feedback_data.normalize_row();
    
    N = 3;
     
    item_item_sim_model = item_item_sim(N); 
    item_item_sim_model.train(feedback_data);
   
    # test. 
    loc_row = [200,   4, 105];
    loc_col = [ 10,  22,   4];
    print 'Prediction:'
    print item_item_sim_model.predict(loc_row, loc_col);
    
