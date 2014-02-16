'''
Created on Feb 16, 2014

@author: jiayu.zhou
'''
from rs.data.daily_watchtime import DailyWatchTimeReader

if __name__ == '__main__':
    #daily_data_file = "/Users/jiayu.zhou/Data/duid-program-watchTime-genre/test_comb/test";
    #daily_data_file_p1 = "/Users/jiayu.zhou/Data/duid-program-watchTime-genre/test_comb/test1";
    #daily_data_file_p2 = "/Users/jiayu.zhou/Data/duid-program-watchTime-genre/test_comb/test2";
    
    daily_data_file = "/Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131210/part";
    daily_data_file_p1 = "/Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131210/part-1";
    daily_data_file_p2 = "/Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131210/part-2";
    
    min_occ_user = 0;
    min_occ_prog = 0;
    
    reader = DailyWatchTimeReader();
    
    #[occur_duid1, occur_pid1, cnt_duid1, cnt_pid1] = reader.read_file_info(daily_data_file);
    
    #[occur_duid2, occur_pid2, cnt_duid2, cnt_pid2] = reader.read_file_info([daily_data_file_p1, daily_data_file_p2]);
    
    data1 = reader.read_file_with_minval(daily_data_file, min_occ_user, min_occ_prog);
    print data1;
    #print data1.get_sparse_matrix().todense();
    
    data2 = reader.read_file_with_minval([daily_data_file_p1, daily_data_file_p2], \
                                          min_occ_user, min_occ_prog);
    print data2;
    #print data2.get_sparse_matrix().todense();