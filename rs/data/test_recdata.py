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
    
    print dataStruct.row_mapping;
    
    print dataStruct.get_sparse_matrix().todense();
    
    
    print '-----------------'
    print 'subsample 3 rows'
    
    [subdata, subidx] = dataStruct.subsample_row(3);
    print subidx;
    
    print subdata.get_sparse_matrix().todense();
    
    print subdata.row_mapping;
    
    
    print '-----------------'
    print 'subsample 50% rows'
    
    [data_split, data_split_comp, selidx_split, selidx_split_comp] = dataStruct.split(0.5);
    
    
    print selidx_split;
    print data_split.get_sparse_matrix().todense();
    print data_split.row_mapping;
    
    print selidx_split_comp;
    print data_split_comp.get_sparse_matrix().todense();
    print data_split_comp.row_mapping;
    
    