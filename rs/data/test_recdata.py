'''
Created on Jan 31, 2014

@author: jiayu.zhou
'''
from rs.data.daily_watchtime import DailyWatchTimeReader


if __name__ == '__main__':
    filename = "../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    reader = DailyWatchTimeReader();
    dataStruct = reader.read_file_with_minval(filename, 7, 1);
    print dataStruct; 
    
    print dataStruct.get_sparse_matrix().todense();
    
    print 'subsample 3 rows'
    
    [subdata, subidx] = dataStruct.subsample_row(3);
    print subidx;
    
    print subdata.get_sparse_matrix().todense();