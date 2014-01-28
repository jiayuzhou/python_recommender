'''
This file creates a configuration file. 

Use this script to write configuration file and avoid directly modifying the 
file, so that there is less chance to mess up with the configuration syntax.  

Created on Jan 27, 2014

@author: jiayu.zhou
'''

from rs.config.config_manager import *; #@UnusedWildImport

if __name__ == '__main__':
    # add data related configurations. 
    config = ConfigParser.RawConfigParser();
    config.add_section(CFG_SEC_DATA);
    config.set(CFG_SEC_DATA,       CFG_DATA_TMPFOLDER,     '/tmp/recsys/rs/data');
    
    # add result related configurations. 
    config.add_section(CFG_SEC_EXPRESULT);
    config.set(CFG_SEC_EXPRESULT, CFG_EXPRESULT_TMPFOLDER, '/tmp/recsys/rs/result')
    
    # add utility related configurations. 
    config.add_section(CFG_SEC_UTILS);
    config.set(CFG_SEC_UTILS,     CFG_UTILS_LOGFOLDER,     '/tmp/recsys/rs/logfiles');
    
    with open('default.cfg', 'wb') as configfile:
        config.write(configfile);
        
    print 'Log file is written successfully.';