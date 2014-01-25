'''
Created on Jan 24, 2014

@author: jiayu.zhou
'''

import sys;
from data.daily_watchtime import DailyWatchTimeReader

if __name__ == '__main__':
    filename = "../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    if len(sys.argv) == 1:
        print 'Use default sample data.'
    else:
        filename = sys.argv[1];
        
    print 'processing file', filename;
    
    reader = DailyWatchTimeReader();
    reader.readLogFile(filename);