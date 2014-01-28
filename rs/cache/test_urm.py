'''
Created on Jan 28, 2014

@author: jiayu.zhou
'''

import numpy as np;
from rs.cache.urm import URM;

if __name__ == '__main__':
    
    a = URM.LoadResource(URM.RTYPE_DATA, 'test001');
    if not a:
        print "not found.";
    
    res = [np.random.rand(3,5), 'test'];
    print res;
    
    URM.SaveResource(URM.RTYPE_DATA, 'test001', res);
    
    a = URM.LoadResource(URM.RTYPE_DATA, 'test001');
    if not a:
        print "not found.";
    
    print res;