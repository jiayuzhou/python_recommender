'''
Management of configurations of the recommendation pipeline. 
Created on Jan 27, 2014

@author: jiayu.zhou
'''

CFG_SEC_DATA            = 'cfg_seg_data';
CFG_DATA_TMPFOLDER      = 'cfg_data_tmpfolder';

CONF_SEC_EXPRESULT      = 'cfg_seg_expresult';
CFG_EXPRESULT_TMPFOLDER = 'cfg_expresult_folder';


class ConfigManager(object):
    '''
    Configuration manager. 
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        