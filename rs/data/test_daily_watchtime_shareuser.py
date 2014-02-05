'''
Test the functionality of the share_row_data function in the rs.data.recdata

Created on Feb 4, 2014

@author: jiayu.zhou
'''

from rs.data.daily_watchtime import DailyWatchTimeReader
from rs.data.recdata import share_row_data;

if __name__ == '__main__':
    
    reader = DailyWatchTimeReader();
    
    filename1 = "../../datasample/agg_duid_pid_watchtime_genre/toy_small_day1";
    filename2 = "../../datasample/agg_duid_pid_watchtime_genre/toy_small_day2";
    
    fb_data1 = reader.read_file_with_minval(filename1, 0, 0);
    fb_data2 = reader.read_file_with_minval(filename2, 0, 0);
    
    print 'Matrix 1'
    print fb_data1.row_mapping;
    print fb_data1.col_mapping;
    print fb_data1.get_sparse_matrix().todense();
    
    print 'Matrix 2'
    print fb_data2.row_mapping;
    print fb_data2.col_mapping;
    print fb_data2.get_sparse_matrix().todense();
    
    # get the share. 
    [fb_data1_share, fb_data2_share] = share_row_data(fb_data1, fb_data2); 
    
    
    print 'Matrix 1 Share'
    print fb_data1_share.row_mapping;
    print fb_data1_share.col_mapping;
    print fb_data1_share.get_sparse_matrix().todense();
    
    print 'Matrix 2 Share'
    print fb_data2_share.row_mapping;
    print fb_data2_share.col_mapping;
    print fb_data2_share.get_sparse_matrix().todense();
    
    