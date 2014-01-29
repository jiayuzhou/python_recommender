'''
Created on Jan 24, 2014

@author: jiayu.zhou
'''

import sys;
from rs.visualization.histgram import histplot; #@UnusedImport


from rs.data.daily_watchtime import DailyWatchTimeReader

if __name__ == '__main__':
    filename = "../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    #filename = "/Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131209/part-r-00000";
    
    if len(sys.argv) == 1:
        print 'Use default sample data.'
    else:
        filename = sys.argv[1];
        
    print 'processing file', filename;
    
    reader = DailyWatchTimeReader();
    #reader.readFile(filename);
    #[mapping_duid, mapping_pid, row, col, data, pggr_pg, pggr_gr] = \
    #    reader.readFileWithMinVal(filename, 5, 5);
    dataStruct = reader.readFileWithMinVal(filename, 1, 1);
    print dataStruct;    
    #print len(pggr_pg);
    #print len(pggr_gr);
    #print len(set(pggr_gr));
    
    #[occur_duid, occur_pid, cnt_duid, cnt_pid] = reader.readFileInfo(filename);
    #print cnt_duid;
    #print cnt_pid;
    
    #histplot(occur_duid.values());
    #histplot(occur_pid.values());
    
    print 'Done';
    
# data: 