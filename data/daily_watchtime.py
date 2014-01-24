'''
Created on Jan 24, 2014

@author: jiayu.zhou
'''        

import csv;        

class DailyWatchTimeReader(object):

    def __init__(self):     
        # the mapping from field to meanings. 
        self.fieldMapping = {'duid':0, 'pid':1, 'watchtime':2, 'genre':3};
        self.fieldDelimiter = '\t';
        self.verbose = True;
    
    def readLogFile(self, filename):
        '''
        This file reads a file, and return 
        1. duid mapping (from duid to an integer, indicates the row number in the sparse matrix); 
        2. pid mapping  (from pid  to an integer, indicates the column number in the sparse matrix);
        3. core sparse matrix. 
        4. a list for genre-program mapping. 
        '''
        
        mapping_duid = {}; # store duid->row# mapping 
        mapping_pid  = {}; # store pid->col# mapping
        
        row  = [];
        col  = [];
        data = [];
        
        with open(filename, 'rb') as csvfile:
            logreader = csv.reader(csvfile, delimiter = self.fieldDelimiter, quotechar = '|');
            for logrow in logreader:
                log_duid      = logrow[self.fieldMapping['duid']];
                log_pid       = logrow[self.fieldMapping['pid']];
                log_watchtime = logrow[self.fieldMapping['watchtime']];
                
                if not (log_duid in mapping_duid):
                    mapping_duid[log_duid] = len(mapping_duid);
                row.append(mapping_duid[log_duid]);
                
                if not (log_pid in mapping_pid):
                    mapping_pid[log_pid]   = len(mapping_pid);
                col.append(mapping_pid[log_pid]);
                
                data.append(log_watchtime);
        
        if (self.verbose):
            print 'Done reading agg log file. '+str(len(data)) + ' elements read'+ \
                ' ( '+str(len(mapping_duid))+' row/user, '+str(len(mapping_pid))+' col/program).';
        
        return [mapping_duid, mapping_pid, row, col, data];
                
    
                