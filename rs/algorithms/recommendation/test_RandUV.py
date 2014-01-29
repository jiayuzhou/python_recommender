'''
This example shows how to load data, build a model instance, train the model 
using loaded data, and finally perform testing. 
 
Created on Jan 29, 2014

@author: jiayu.zhou
'''

from rs.data.daily_watchtime import DailyWatchTimeReader
from rs.algorithms.recommendation.RandUV import RandUV

if __name__ == '__main__':
    filename = "../../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    # load data. 
    reader = DailyWatchTimeReader();
    feedback_data = reader.read_file_with_minval(filename, 1, 1);
    
    # build model with 3 latent factors. 
    rand_model = RandUV(3, verbose = True);
    rand_model.train(feedback_data);
    
    # test. 
    loc_row = [200,   4, 105];
    loc_col = [ 10,  22,   4];
    print 'Prediction:'
    print rand_model.predict(loc_row, loc_col);
    
    