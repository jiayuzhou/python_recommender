'''
Created on Jan 24, 2014

@author: jiayu.zhou
'''        

import csv;        
from rs.cache.urm import URM;
from rs.utils.log import Logger;
from rs.data.recdata import FeedbackData;

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
                    Logger.Log(str(lineNum) + ' lines read.');
                    
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
        
        res_str  = 'DWT_RFWMV[' + filename + '][MIN DUID' + str(min_duid) + '][MIN PID' + str(min_pid) +']';
        
        # We check if the current resource is available. If not then load from test data and save resource.  
        if not URM.CheckResource(URM.RTYPE_DATA, res_str): 
        
            Logger.Log('Computing data information...');
            [occur_duid, occur_pid, _, _] = self.readFileInfo(filename);
            print str(len(occur_duid)), 'devices', str(len(occur_pid)), 'programs';
            
            Logger.Log('Generating filtering indices...');
            duidlist = set([sel_duid for sel_duid, sel_duidcnt in occur_duid.iteritems() if sel_duidcnt > min_duid]);
            pidlist  = set([sel_pid  for sel_pid,  sel_pidcnt  in occur_pid.iteritems()  if sel_pidcnt  > min_pid]);
            print 'After filtering [MIN_DUID',str(min_duid), ' MIN_PID:', str(min_pid),']:',\
                str(len(occur_duid)), 'devices', str(len(occur_pid)), 'programs';
            
            # read the raw data file with the list.
            [mapping_duid, mapping_pid, row, col, data, pggr_pg, pggr_gr] \
                = self.readFileWithIDList(filename, duidlist, pidlist);
                
            Logger.Log('readFileWithMinVal process completed.');
            
            result = FeedbackData(row, col, data, len(mapping_duid), len(mapping_pid),\
                    mapping_duid, mapping_pid, {'pggr_pg': pggr_pg, 'pggr_gr': pggr_gr});
            
            # save computed results to resource cache. 
            URM.SaveResource(URM.RTYPE_DATA, res_str, result);    
            return result;
        else:
            return URM.LoadResource(URM.RTYPE_DATA, res_str);
    
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
        pggr_pg = [];
        pggr_gr = [];
        
        visited_program_list = set([]);
        
        lineNum = 0;
        with open(filename, 'rb') as csvfile:
            logreader = csv.reader(csvfile, delimiter = self.fieldDelimiter, quotechar = '|');
            for logrow in logreader:
                log_duid      = logrow[self.fieldMapping['duid']];
                log_pid       = logrow[self.fieldMapping['pid']];
                log_pg_gr     = logrow[self.fieldMapping['genre']];
                
                ## we need both duid and pid are in the list. 
                if (log_duid in duidlist) and (log_pid in pidlist):
                
                    log_watchtime = logrow[self.fieldMapping['watchtime']];
                    
                    if not (log_duid in mapping_duid):
                        mapping_duid[log_duid] = len(mapping_duid);
                    row.append(mapping_duid[log_duid]);
                    
                    if not (log_pid in mapping_pid):
                        mapping_pid[log_pid]   = len(mapping_pid);
                    col.append(mapping_pid[log_pid]);
                    
                    # store program - genre mappings. 
                    for pg_gr in log_pg_gr.split(','):
                        if not pg_gr:
                            Logger.Log('Empty genre information for program '+log_pid, Logger.MSG_CATEGORY_DATA);
                            continue;
                        if not mapping_pid[log_pid] in visited_program_list:
                            pggr_pg.append(mapping_pid[log_pid]);
                            pggr_gr.append(int(pg_gr));
                            visited_program_list.add(mapping_pid[log_pid]);
                    
                    data.append(log_watchtime);
                
                lineNum+=1;
                
                if self.verbose and (lineNum%self.display == 0):
                    print str(lineNum), ' lines read.';
                    
        if (self.verbose):
            Logger.Log('Done reading agg log file. '+str(len(data)) + ' elements read'+ \
                ' ( '+str(len(mapping_duid))+' row/user, '+str(len(mapping_pid))+' col/program).');
        
        return [mapping_duid, mapping_pid, row, col, data, pggr_pg, pggr_gr];
        
    
    def readFile(self, filename):
        '''
        This file reads an aggregated file, and return 
        1. duid mapping (from duid to an integer, indicates the row number in the sparse matrix); 
        2. pid mapping  (from pid  to an integer, indicates the column number in the sparse matrix);
        3. core sparse matrix. 
        4. a list for genre-program mapping. 
        
        Note: this method pops data for ALL users. 
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
                    Logger.Log(str(lineNum) + ' lines read.');
        
        if (self.verbose):
            Logger.Log('Done reading agg log file. '+str(len(data)) + ' elements read'+ \
                ' ( '+str(len(mapping_duid))+' row/user, '+str(len(mapping_pid))+' col/program).');
        
        return [mapping_duid, mapping_pid, row, col, data];
                
    
                