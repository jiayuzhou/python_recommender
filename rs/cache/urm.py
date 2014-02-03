'''
Universal Resource Manager. 

Created on Jan 27, 2014

@author: jiayu.zhou
'''


import os; #@UnusedImport
import hashlib;
import cPickle as pickle;
from rs.config.config_manager import *; #@UnusedWildImport
from rs.utils.log import Logger;


class URM(object):
    '''
    Universal resource manager (URM). URM should be used 
    in a factory access to avoid duplicated instance.
    
    To request a resource in cache there are two parameters:
    1. the resource_type: e.g., UCManager.RTYPE_DATA, UCManager.RTYPE_RESULT;
    2. the unique_resource_str: a string that uniquely identify the resource. 
             The resource string will be hashed using SHA224 and we assume that 
             there is no collision. 
    '''
    
    RTYPE_DATA   = 'rt_data';
    
    RTYPE_RESULT = 'rt_result';
    
    # save experimental settings (splittings, data finger-prints).  
    RTYPE_EXPCFG = 'rt_exp_config';

    _instance = None;

    DefaultCacheFolder = '/tmp/recsys'; #default cache folder


    def __init__(self, cache_folder = None, use_cache = None):
        '''
        Create a cache manager with the specified cache location. 
        '''
        
        # if cache is system is used.
        # if turned off then save/load/check resource will not do anything.  
        config_use_cache = ConfigManager.GetBoolean(CFG_SEC_UTILS, CFG_UTILS_USECACHE);
        self.use_cache = config_use_cache if use_cache is None else use_cache; 
        if not self.use_cache:
            Logger.Log("URM is turned off. All cached resources are not available.", \
                       Logger.MSG_CATEGORY_SYSTEM)
        
        # set up the directory for cache.  
        config_cache_folder = ConfigManager.Get(CFG_SEC_UTILS, CFG_UTILS_CACHEFOLDER);
        self.cache_folder = config_cache_folder if cache_folder is None else cache_folder;
        
        # create directory if it does not exist
        self.cacheLocation = self.cache_folder;
        if not os.path.exists(self.cacheLocation):
            os.makedirs(self.cacheLocation);
    
    def universal_resrouce_location(self, resource_type, unique_resource_str, sub_folder = None):
        '''
        Get the FULL PATH and NAME of file.
        resource_type_hash($unique_resource_str$);
        '''
        
        # construct the file name (without path).
        filename = resource_type + '_' + hashlib.sha224(unique_resource_str).hexdigest();
        
        if not sub_folder:
            
            res_loc  = self.cache_folder + '/' + resource_type;
            if not os.path.exists(res_loc):
                os.makedirs(res_loc);
            out = res_loc + '/' + filename;
        else:
            res_loc  = self.cache_folder + '/' + resource_type + '/' + sub_folder;
            if not os.path.exists(res_loc):
                os.makedirs(res_loc);
            out = res_loc + '/' + filename;
        return out;
    
    @classmethod
    def CheckResource(cls, resource_type, unique_resource_str, sub_folder = None):
        # return false if cache is turned off. 
        if not cls.GetInstance().use_cache:
            return False;
        
        # generate full resource address.
        url = cls.GetInstance().universal_resrouce_location(resource_type, unique_resource_str, sub_folder);
        return os.path.isfile(url);
        
    @classmethod
    def LoadResource(cls, resource_type, unique_resource_str, sub_folder = None):
        '''
        Get cached resource. The method firstly use resource type and unique resource string to 
        construct a file name with full path.  
        '''
        if not cls.GetInstance().use_cache:
            return None;
        
        url = cls.GetInstance().universal_resrouce_location(resource_type, unique_resource_str, sub_folder);
        
        if not os.path.isfile(url):
            Logger.Log("Resource["+ resource_type +"]["+ unique_resource_str +"] not found.");
            return None;
        else:
            Logger.Log("Loading resource ["+ resource_type +"]["+ unique_resource_str +"] ...");
            return pickle.load(open(url, "rb"));
            Logger.Log("Resource["+ resource_type +"]["+ unique_resource_str +"] loaded.");
        
    @classmethod
    def SaveResource(cls, resource_type, unique_resource_str, content, sub_folder = None):
        '''
        Put resource in cache. 
        '''
        if not cls.GetInstance().use_cache:
            return;
        
        url = cls.GetInstance().universal_resrouce_location(resource_type, unique_resource_str, sub_folder);
        
        if not os.path.isfile(url):
            Logger.Log("Saving resource ["+ resource_type +"]["+ unique_resource_str +"]...");
            pickle.dump(content, open(url,"wb"));
            Logger.Log("Resource["+ resource_type +"]["+ unique_resource_str +"] saved.");
        else:
            Logger.Log("Resource["+ resource_type +"]["+ unique_resource_str +"] found. Save skipped.");
    
    @classmethod
    def GetInstance(cls):
        '''
        Get the factory/global instance of UCManager.
        '''
        if not cls._instance:
            cls._instance = URM();
        return cls._instance;
            