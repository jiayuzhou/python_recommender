'''
Created on Apr 29, 2014

@author: jiayu.zhou
'''
#from StringIO import StringIO
from rs.data.utility_data import UtilityDataReader
from rs.data.recdata import FeedbackData
from copy import deepcopy

if __name__ == '__main__':
    
    #txt = "U1\tD1\t44\n"+"U2\tD2\t10\n"+"U2\tD1\t20\n"    
    #for line in StringIO(txt):
    #    print line
    
    
    
    data_file = '../../datasample/agg_duid_pid_watchtime_genre/toy_small_day1'
    
    reader = UtilityDataReader();
    feedback = reader.readFile(data_file);
    print feedback.col_mapping
    
    col_feature_map= {};
    col_feature_map['P0001'] = 'feature1'
    col_feature_map['P0002'] = 'feature2'
    col_feature_map['P0003'] = 'feature3'
    col_feature_map['P0004'] = 'feature4'
    
    feedback2 = deepcopy(feedback)
    
    feedback.attach_col_feature(col_feature_map)
    
    print feedback.meta[FeedbackData.METAKEY_COL_FEATURE]
    
    print feedback2
    print feedback
    # successful attached features.  