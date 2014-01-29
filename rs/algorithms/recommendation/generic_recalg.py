'''
Created on Jan 29, 2014

@author: jiayu.zhou
'''



class CFAlg(object):
    '''
    Generic collaborative filtering algorithm. The algorithms in the interface 
    should have a train method that takes an input of the type:
        rs.data.recdata.FeedbackData
    And also a predict function takes a list of positions (that previous unknown).
    '''

    def train(self, feedbackData):
        '''
        The feedback data should use the following data structure:
        >> from rs.data.recdata import FeedbackData;
        '''
        raise NotImplementedError("Interface method.");
    
    def predict(self, row, col):
        '''
        Predict values in the tuples (row[i], col[i]). Return an array/list of predictions.  
        '''
        raise NotImplementedError("Interface method.");