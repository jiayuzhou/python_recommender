'''
Created on Jan 28, 2014

@author: jiayu.zhou
'''

class GenericData(object):
    '''
    A generic data set. 
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        pass;
    
    def split(self, percentage):
        '''
        Provide a random splitting of data. The return should be 
        two data objects of the same data class. 
        '''
        raise NotImplementedError("Interface method.");
    
    def fold(self, fold_num, total_fold):
        '''
        Return one fold of the data out of a n fold splitting.  
        The splitting must be deterministic. 
        
        e.g., self.fold(1, 5) must be the same for each call. 
        '''
        raise NotImplementedError("Interface method.");