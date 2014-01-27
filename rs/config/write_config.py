'''
This file creates a configuration file. 
Created on Jan 27, 2014

@author: jiayu.zhou
'''

import ConfigParser;
from rs.config.config_manager import *;

if __name__ == '__main__':
    config = ConfigParser.RawConfigParser();
    config.add_section(CFG_SEC_DATA);
    config.set(CFG_SEC_DATA,       CFG_DATA_TMPFOLDER,      '/tmp/recsys/data');
    
    config.add_section(CONF_SEC_EXPRESULT);
    config.set(CONF_SEC_EXPRESULT, CFG_EXPRESULT_TMPFOLDER, '/tmp/recsys/result')
    
    with open('example.cfg', 'wb') as configfile:
        config.write(configfile);