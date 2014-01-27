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
        self.display = 100000; # gives output after this number of lines are read. 
    
    def readFileInfo(self, filename):
        '''
        This file reads an aggregated file and get summary (occurrences) 
        for program and device. This information can be used to filtered 
        out programs/devices later. 
        '''
        
        occur_duid = {}; 
        occur_pid  = {}; 
        #ttime_duid = {};
        #ttime_pid  = {};
        
        lineNum = 0;
        with open(filename, 'rb') as csvfile:
            logreader = csv.reader(csvfile, delimiter = self.fieldDelimiter, quotechar = '|');
            for logrow in logreader:
                log_duid      = logrow[self.fieldMapping['duid']];
                log_pid       = logrow[self.fieldMapping['pid']];
                
                if not (log_duid in occur_duid):
                    occur_duid[log_duid]  = 1;
                else:
                    occur_duid[log_duid] += 1;
                
                if not (log_pid in occur_pid):
                    occur_pid[log_pid]   = 1;
                else:
                    occur_pid[log_pid]  += 1;
        
                lineNum+=1;
                if self.verbose and (lineNum % self.display == 0):
                    print str(lineNum), ' lines read.';
                    
        # count occurrence into bins. 
        cnt_duid = {}; # cnt_duid[number of occurrence] = number of duid with specific occurrence. 
        for val in occur_duid.values():
            if not (val in cnt_duid):
                cnt_duid[val]  = 1;
            else:
                cnt_duid[val] += 1;
        
        cnt_pid = {}; # cnt_duid[number of occurrence] = number of duid with specific occurrence. 
        for val in occur_pid.values():
            if not (val in cnt_pid):
                cnt_pid[val]  = 1;
            else:
                cnt_pid[val] += 1;
        
        return [occur_duid, occur_pid, cnt_duid, cnt_pid];
    
    
    def readFileWithMinVal(self, filename, min_duid, min_pid):
        '''
        This method first goes through the data once, and filter out 
        the device and program that has occurrences below specified values. 
        '''
        print 'Computing data information...';
        [occur_duid, occur_pid, _, _] = self.readFileInfo(filename);
        print str(len(occur_duid)), 'devices', str(len(occur_pid)), 'programs';
        
        print 'Filtering...'
        duidlist = set([sel_duid for sel_duid, sel_duidcnt in occur_duid.iteritems() if sel_duidcnt > min_duid]);
        pidlist  = set([sel_pid  for sel_pid,  sel_pidcnt  in occur_pid.iteritems()  if sel_pidcnt  > min_pid]);
        
        print 'After filtering [MIN_DUID',str(min_duid), ' MIN_PID:', str(min_pid),']:',\
            str(len(occur_duid)), 'devices', str(len(occur_pid)), 'programs';
        [mapping_duid, mapping_pid, row, col, data] = self.readFileWithIDList(filename, duidlist, pidlist);
        print 'Done';
        return [mapping_duid, mapping_pid, row, col, data];
    
    
    def readFileWithIDList(self, filename, duidlist, pidlist):
        '''
        This file reads an aggregated file. 
        The file only include the specified duid and pid.  
        '''
        
        mapping_duid = {}; # store duid->row# mapping 
        mapping_pid  = {}; # store pid->col# mapping
        
        row  = [];
        col  = [];
        data = [];
        
        lineNum = 0;
        with open(filename, 'rb') as csvfile:
            logreader = csv.reader(csvfile, delimiter = self.fieldDelimiter, quotechar = '|');
            for logrow in logreader:
                log_duid      = logrow[self.fieldMapping['duid']];
                log_pid       = logrow[self.fieldMapping['pid']];
                
                ## we need both duid and pid are in the list. 
                if (log_duid in duidlist) and (log_pid in pidlist):
                
                    log_watchtime = logrow[self.fieldMapping['watchtime']];
                    
                    if not (log_duid in mapping_duid):
                        mapping_duid[log_duid] = len(mapping_duid);
                    row.append(mapping_duid[log_duid]);
                    
                    if not (log_pid in mapping_pid):
                        mapping_pid[log_pid]   = len(mapping_pid);
                    col.append(mapping_pid[log_pid]);
                    
                    data.append(log_watchtime);
                
                lineNum+=1;
                #print str(lineNum), ' lines read.';
                
                if self.verbose and (lineNum%self.display == 0):
                    print str(lineNum), ' lines read.';
                    
        if (self.verbose):
            print 'Done reading agg log file. '+str(len(data)) + ' elements read'+ \
                ' ( '+str(len(mapping_duid))+' row/user, '+str(len(mapping_pid))+' col/program).';
        
        return [mapping_duid, mapping_pid, row, col, data];
        
    
    def readFile(self, filename):
        '''
        This file reads an aggregated file, and return 
        1. duid mapping (from duid to an integer, indicates the row number in the sparse matrix); 
        2. pid mapping  (from pid  to an integer, indicates the column number in the sparse matrix);
        3. core sparse matrix. 
        4. a list for genre-program mapping. 
        
        Note: this method pops data for users. 
        '''
        
        mapping_duid = {}; # store duid->row# mapping 
        mapping_pid  = {}; # store pid->col# mapping
        
        row  = [];
        col  = [];
        data = [];
        
        lineNum = 0;
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
                
                lineNum+=1;
                if self.verbose and (lineNum%self.display == 0):
                    print str(lineNum), ' lines read.';
                    
        
        if (self.verbose):
            print 'Done reading agg log file. '+str(len(data)) + ' elements read'+ \
                ' ( '+str(len(mapping_duid))+' row/user, '+str(len(mapping_pid))+' col/program).';
        
        return [mapping_duid, mapping_pid, row, col, data];
                
    
                