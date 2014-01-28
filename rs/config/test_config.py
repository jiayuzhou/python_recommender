'''
Created on Jan 27, 2014

@author: jiayu.zhou
'''

from rs.config.config_manager import *; #@UnusedWildImport

if __name__ == '__main__':
    a = ConfigManager.GetInstance();
    b = ConfigManager.GetInstance();
    
    # check 
    print id(a);
    print id(b);
    
    print ConfigManager.Get(CFG_SEC_DATA, CFG_DATA_TMPFOLDER);
    print ConfigManager.Get(CFG_SEC_UTILS, CFG_UTILS_LOGFOLDER);