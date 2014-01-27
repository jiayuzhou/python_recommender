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
    classdocs
    '''

    _instance = None;

    DefaultCacheFolder = '/tmp/recsys'; #default cache folder

    def __new__(cls, *args, **kwargs):
        '''
        Singleton allow only one instance to be created. 
        Pattern reference:
        http://stackoverflow.com/questions/42558/python-and-the-singleton-pattern
        '''
        if not cls._instance:
            cls._instance = super(UCManager, cls).__new__(
                                cls, *args, **kwargs)
        return cls._instance;

    def __init__(self, cache_folder = UCManager.DefaultCacheFolder):
        '''
        Create a cache manager with the specified cache location. 
        '''
        self.cacheLocation = cache_folder;
        if not os.path.exists(self.cacheLocation):
            os.makedirs(self.cacheLocation);
    
    def universalFile(self, resource_type, unique_resource_str):
        
        pass;
    
    def getCache(self, resource_type, unique_resource_str):
        
        pass;
    
    def putCache(self, resource_type, unique_resource_str, content):
        
        pass;
    
    @classmethod
    def getInstance(cls):
        return cls._instance;
            