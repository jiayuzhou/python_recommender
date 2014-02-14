'''
Created on Feb 13, 2014

@author: jiayu.zhou
'''

import scipy.io as sio;

from rs.data.daily_watchtime import DailyWatchTimeReader


if __name__ == '__main__':
    daily_data_file = "/Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131209/part-r-00000";
    reader = DailyWatchTimeReader();
    data = reader.read_file_with_minval(daily_data_file, 1, 1);
    
    data_mat = data.get_sparse_matrix();
    
    sio.savemat("/Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131209.mat", {'data': data_mat})
    
    print 'Done';
    
    