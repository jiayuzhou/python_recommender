'''
Created on Jan 27, 2014

@author: jiayu.zhou
'''
import datetime;
import logging;
from rs.config.config_manager import *; #@UnusedWildImport
from rs.utils.dir import check_create_dir;

class Logger(object):
    '''
    Logging
    '''
    
    _instance = None;
    
    @staticmethod
    def GetTimeString():
        '''
        A string representing current local time in a string that is good for file name.
        '''
        localdatetime = datetime.datetime.now();
        return localdatetime.strftime("%Y-%m-%d-%H%MZ");
    
    def __init__(self, logfolder = None, display_level = 0):
        '''
        Constructor
        >>logfolder - The folder used to store log files. The default folder location 
                      is given in the configuration CFG_SEC_UTILS:CFG_UTILS_LOGFOLDER. 
        >>display_level - display level. E.g., if set to 1 then messages below level 1 
                          will not be displayed (but will still in log file).
        '''
        configDir = ConfigManager.Get(CFG_SEC_UTILS, CFG_UTILS_LOGFOLDER);
        
        self.logfolder = configDir if logfolder is None else logfolder;
        
        check_create_dir(logfolder, True); # create folder for log file. 
        
        self.logfile = '%s/%s.log' % (self.logfolder, Logger.GetTimeString() );
        logging.basicConfig(filename=self.logfile,level=logging.INFO)
        
        self.display_level = display_level;
        
        print 'Initialize log file:', self.logfile;
        logging.info('I told you so')
        
        
        
        
    def __exit__(self, type, value, traceback):
        pass; #clean files. 
    
    
    def _log(self, message, category, display_level):
        '''
        
        '''
        message_str = '['+ Logger.GetTimeString() + '][' + category + ']'\
                      + message;
        
        logging.info(message_str);
        
        if display_level >= self.display_level:
            print message_str;
    
    @classmethod
    def GetInstance(cls):
        '''
        Factory. 
        '''
        if not cls._instance:
            cls._instance = Logger();
        return cls._instance;
    
    @classmethod
    def Log(cls, message, category = 'DEFAULT', display_level = 0):
        cls.GetInstance()._log(message, category,display_level);
        
        