'''
Management of configurations of the recommendation pipeline. 
Created on Jan 27, 2014

@author: jiayu.zhou
'''
import ConfigParser;
import os;

# More about ConfigParser:
# http://docs.python.org/2/library/configparser.html
       
# constant field. 
CFG_SEC_DATA            = 'cfg_seg_data';
CFG_DATA_TMPFOLDER      = 'cfg_data_tmpfolder';

CFG_SEC_EXPRESULT      = 'cfg_seg_expresult';
CFG_EXPRESULT_TMPFOLDER = 'cfg_expresult_folder';

CFG_SEC_UTILS          = 'cfg_seg_utils';
CFG_UTILS_LOGFOLDER    = 'cfg_utils_logfolder';

class ConfigManager(object):
    '''
    Configuration manager. 
    This is a factory class and to access config use GetInstance().
    '''
    
    _instance = None; #factory. 
    
    DEFAULT_CFG_FILE = os.path.dirname(os.path.realpath(__file__)) + '/' + 'default.cfg';

    def __init__(self, cfg_file = None):
        '''
        Constructor
        '''
        self.cfg_file = self.DEFAULT_CFG_FILE if cfg_file is None else cfg_file;
        
        print 'Initialize configuration...';
        self.config = ConfigParser.RawConfigParser();
        self.config.read(self.cfg_file);
        
        print 'Successfully read config from file:', self.cfg_file;
    
    @classmethod
    def SwitchConfig(cls, filename):
        '''
        Switch to another config file. 
        '''
        # if filename has changed, we recreate the config instance. 
        if cls._instance.cfg_file is not filename:
            cls._instance = ConfigManager(filename);
    
    @classmethod
    def GetInstance(cls):
        '''
        Get the factory/global instance of ConfigManager.
        '''
        if not cls._instance:
            cls._instance = ConfigManager();
        return cls._instance;
    
    @classmethod
    def Get(cls, section, option):
        return cls.GetInstance().config.get(section, option);
        