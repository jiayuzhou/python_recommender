'''
Created on Jan 27, 2014

@author: jiayu.zhou
'''
import os;
import rs.utils.log;

def check_create_dir(dirstr, silent = False):
    '''
    Check if current directory exists, if not create it. 
    '''
    if not os.path.exists(dirstr):
        os.makedirs(dirstr);   
        
    if not silent:
        rs.utils.log.Logger.Log("Created folder: "+ dirstr);