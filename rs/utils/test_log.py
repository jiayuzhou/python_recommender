'''
Created on Jan 27, 2014

@author: jiayu.zhou
'''
from rs.utils.log import Logger

if __name__ == '__main__':
    logger = Logger('./logs');
    logger._log('Hello world', 'TEST', 0);
    logger._log('The second line', 'TEST', 0 );
    logger._log('Something else.', 'TEST', 0);