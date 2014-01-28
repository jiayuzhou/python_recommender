'''
Universal Cache Manager. 


Created on Jan 27, 2014

@author: jiayu.zhou
'''


import os;

RTYPE_DATA   = 'rt_data';
RTYPE_RESULT = 'rt_result';



class UCManager(object):
    '''
    Universal cache/resource manager. 
    '''

    _instance = None;

    DefaultCacheFolder = '/tmp/recsys'; #default cache folder


    def __init__(self, cache_folder = None):
        '''
        Create a cache manager with the specified cache location. 
        '''
        self.cacheLocation = cache_folder;
        if not os.path.exists(self.cacheLocation):
            os.makedirs(self.cacheLocation);
    
    @classmethod
    def universalFile(cls, resource_type, unique_resource_str):
        '''
        Get the representation of file.
        '''
        
        pass;
    
    @classmethod
    def getCache(cls, resource_type, unique_resource_str):
        
        pass;
    
    @classmethod
    def putCache(cls, resource_type, unique_resource_str, content):
        
        pass;
    
    @classmethod
    def GetInstance(cls):
        '''
        Get the factory/global instance of UCManager.
        '''
        if not cls._instance:
            cls._instance = UCManager();
        return cls._instance;
            