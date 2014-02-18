'''
Created on Jan 24, 2014

@author: jiayu.zhou
'''        

import csv;        
from rs.cache.urm import URM;
from rs.utils.log import Logger;
from rs.data.recdata import FeedbackData;
import random;

class DailyWatchTimeReader(object):

    def __init__(self):     
        # the mapping from field to meanings. 
        self.fieldMapping = {'duid':0, 'pid':1, 'watchtime':2, 'genre':3};
        self.fieldDelimiter = '\t';
        self.verbose = True;
        self.display = 100000; # gives output after this number of lines are read. 
    
    def read_file_info(self, filename):
        '''
        This file reads an aggregated file and get summary (occurrences) 
        for program and device. This information can be used to filtered 
        out programs/devices later. 
        '''
        
        occur_duid = {}; 
        occur_pid  = {}; 
        #ttime_duid = {};
        #ttime_pid  = {};
        
        # turn a single file into a file list. 
        if not isinstance(filename, list):
            filename_arr = [filename];
        else:
            filename_arr = filename;  
        
        lineNum = 0;
        
        for filename in filename_arr:
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
            
                    lineNum += 1;
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
    
    
    def read_file_with_minval(self, filename, min_duid, min_pid, num_duid = None, num_pid = None, rand_seed = 1):
        '''
        This method first goes through the data once, and filter out 
        the device and program that has occurrences below specified values. 
        
        Parameters
        ----------
        filename: a string consists of the file name and location of the data file to be read.
        min_duid: a positive integer. the minimum occurrence of a device for the device to be included.
        min_pid:  a positive integer. the minimum occurrence of a program for the program to be included. 
         
        Returns
        ----------
        result: a FeedbackData data structure constructed from the data file. In the result there is also 
                a genre-program mapping data (result.meta['pggr_pg'][i], meta['pggr_pr'][i]) indicates that 
                the program at result.meta['pggr_pg'][i] is marked by genre at meta['pggr_pr'][i]. The genre 
                mapping is in R:/Data/Rovi/genre.csv, and a vintage copy is also kept in datasample/Rovi folder.
        '''
        
        if num_duid is None and num_pid is None:
            subsample = False;
            res_str  = 'DWT_RFWMV[' + str(filename) + '][MIN DUID' + str(min_duid) + '][MIN PID' + str(min_pid) +']';
        elif num_duid is not None and num_pid is not None:
            subsample = True;
            res_str  = 'DWT_RFWMV[' + str(filename) + '][MIN DUID' + str(min_duid) + '][MIN PID' + str(min_pid) +']'\
                            + '[NUM DUID' + str(num_duid) + ']' + '[NUM PID' + str(num_pid) + ']';
        else:
            raise ValueError('num_duid and num_pid should be both set or both use default');
        
        
        
        
        # We check if the current resource is available. If not then load from test data and save resource.  
        if not URM.CheckResource(URM.RTYPE_DATA, res_str): 
        
            Logger.Log('Computing data information...');
            [occur_duid, occur_pid, _, _] = self.read_file_info(filename);
            print str(len(occur_duid)), 'devices', str(len(occur_pid)), 'programs';
            
            Logger.Log('Generating filtering indices...');
            duidlist = [sel_duid for sel_duid, sel_duidcnt in occur_duid.iteritems() if sel_duidcnt > min_duid];
            pidlist  = [sel_pid  for sel_pid,  sel_pidcnt  in occur_pid.iteritems()  if sel_pidcnt  > min_pid];
            
            print 'After filtering [MIN_DUID',str(min_duid), ' MIN_PID:', str(min_pid),']:',\
                str(len(occur_duid)), 'devices', str(len(occur_pid)), 'programs';
            
            # perform random sampling.
            if subsample:
                random.seed(rand_seed);
                if len(duidlist) > num_duid:
                    # subsample DUID;
                    random.shuffle(duidlist);
                    duidlist = duidlist[:num_duid];
                
                if len(pidlist)  > num_pid:
                    # subsample PID;
                    random.shuffle(pidlist);
                    pidlist  = pidlist[:num_pid];
            
            duidlist = set(duidlist);
            pidlist  = set(pidlist);
            
            # read the raw data file with the list.
            [mapping_duid, mapping_pid, row, col, data, pggr_pg, pggr_gr] \
                = self.read_file_with_id_list(filename, duidlist, pidlist);
                
            Logger.Log('read_file_with_minval process completed.');
            
            result = FeedbackData(row, col, data, len(mapping_duid), len(mapping_pid),\
                    mapping_duid, mapping_pid, {'pggr_pg': pggr_pg, 'pggr_gr': pggr_gr});
            
            # save computed results to resource cache. 
            URM.SaveResource(URM.RTYPE_DATA, res_str, result);    
            return result;
        else:
            return URM.LoadResource(URM.RTYPE_DATA, res_str);
    
    def read_file_with_id_list(self, filename, duidlist, pidlist, ignore_prog_without_genre = True):
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
        
        # turn a single file into a file list. 
        if not isinstance(filename, list):
            filename_arr = [filename];
        else:
            filename_arr = filename;  
        
        for filename in filename_arr:
            with open(filename, 'rb') as csvfile:
                logreader = csv.reader(csvfile, delimiter = self.fieldDelimiter, quotechar = '|');
                for logrow in logreader:
                    log_duid      = logrow[self.fieldMapping['duid']];
                    log_pid       = logrow[self.fieldMapping['pid']];
                    log_pg_gr     = logrow[self.fieldMapping['genre']].strip();
                    
                    if not log_pg_gr:
                        Logger.Log('Empty genre information for program '+log_pid, Logger.MSG_CATEGORY_DATA);
                        if ignore_prog_without_genre:
                            # ignore records whose program has no genre information. 
                            continue;
                    
                    ## we need both duid and pid are in the list. 
                    if (log_duid in duidlist) and (log_pid in pidlist):
                    
                        log_watchtime = int(logrow[self.fieldMapping['watchtime']]);
                        
                        if not (log_duid in mapping_duid):
                            mapping_duid[log_duid] = len(mapping_duid);
                        row.append(mapping_duid[log_duid]);
                        
                        if not (log_pid in mapping_pid):
                            mapping_pid[log_pid]   = len(mapping_pid);
                        col.append(mapping_pid[log_pid]);
                        
                        # store program - genre mappings, for programs that were not visited before.
                        if not mapping_pid[log_pid] in visited_program_list: 
                            for pg_gr in log_pg_gr.split(','):
                                if not pg_gr:
                                    continue;
                                
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
                log_watchtime = int(logrow[self.fieldMapping['watchtime']]);
                
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
                
    
                