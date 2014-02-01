# This the daily log entity for a ROVI daily log file. 
# Created: Jiayu Zhou, Jan 20, 2014. 

import csv;
from jiayuUtils import combineSetDictList;

class DailyLog:
    """ Daily Log """

    def __init__(self):
        self.content = [];
        self.progDic = {};
        self.genreDic = {};
    
    @staticmethod
    def createFromFile(fileName):
        logObj = DailyLog();
        with open(fileName, 'rb') as csvfile:
            logreader = csv.reader(csvfile, delimiter = '\t', quotechar = '|')
            for row in logreader:
                #print ','.join([row[1], row[7], row[9]]);
                #print ','.join(row);
                entry = LogEntry(row);
                logObj.content.append(entry);
                # add to program dictionary 
                entryId = entry.chprId; 
                if not (entryId in logObj.progDic):
                    logObj.progDic[entryId] = [];
                logObj.progDic[entryId].append(entry);
                # add genre to genre list. 
                for entryGenreId in entry.genreIdArr:
                    if (entryGenreId == ''): # skip entries without genre info. 
                        continue; 
                    if not (entryGenreId in logObj.genreDic):
                        logObj.genreDic[entryGenreId] = [];
                    logObj.genreDic[entryGenreId].append(entry);
                
        return logObj;

    def length(self):
        return len(self.content);

    def mergeLog(self, anotherLog):
        # get a merged log.
        mergedLog = DailyLog();
        mergedLog.content  = self.content + anotherLog.content;
        #mergedLog.progDic  = dict(self.progDic.items() + anotherLog.progDic.items());
        mergedLog.progDic  = combineSetDictList([self.progDic, anotherLog.progDic])
        #mergedLog.genreDic = dict(self.genreDic.items() + anotherLog.genreDic.items());
        mergedLog.genreDic = combineSetDictList([self.genreDic, anotherLog.genreDic])
        return mergedLog;

    @staticmethod 
    def mergeLogs(setOfLogs):
        mergeLog = DailyLog();
        mergeLog.content  = [cont for log in setOfLogs for cont in log.content];
        mergeLog.progDic  = combineSetDictList([log.progDic  for log in setOfLogs]);
        mergeLog.genreDic = combineSetDictList([log.genreDic for log in setOfLogs]);
        return mergeLog;

    def overlap(self, anotherLog):
        # compute the overlap of two logs. 
        return len(self.progDic.keys()) + len(anotherLog.progDic.keys()) - len(set(self.progDic.keys() + anotherLog.progDic.keys()));

    def showOverlapRatio(self, anotherLog):
        numInter = len(self.progDic.keys()) + len(anotherLog.progDic.keys()) - len(set(self.progDic.keys() + anotherLog.progDic.keys()));
        numUnion  = len(set(self.progDic.keys() + anotherLog.progDic.keys()));
        ratio = float(numInter)/float(numUnion);
        print 'Overlap/Union: ' + str(numInter) + '/' + str(numUnion) + ' (' + str(ratio) + ')'; 
        
class LogEntry:
    """ entry object used in DailyLog """

    def __init__(self, logCSVEntry):
        """ logCSVEntry is a row parsed by csv.reader """ 
        self.channelId   = logCSVEntry[0];
        self.channelName = logCSVEntry[1];
        self.progId      = logCSVEntry[3];
        self.progName    = logCSVEntry[7];
        self.genreIdArr  = logCSVEntry[9].strip().split(','); # the manually generated unique id for the program.

        self.chprId      = self.channelName + ':' + self.progName;

        ## check invalid genreId. 
        #for genreId in self.genreIdArr:
        #    if (genreId == ''):
        #        print logCSVEntry;
        
    def __str__(self):
        return '[' + self.progName+ ':'.join(self.genreIdArr) + ']';
    
    def __repr__(self):
        return '[' + self.progName + ':'+ ','.join(self.genreIdArr) + ']';

class GenreMap:
    genreMapFile = './genre.csv'; # default. 
    genreMap = {};
    
    @staticmethod
    def GetMap():
        if (not GenreMap.genreMap):        
            with open ('./genre.csv', 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter = ',');
                for row in reader:
                    GenreMap.genreMap[row[0]] = row[1];
            print 'Genre map loaded successfully from [' + GenreMap.genreMapFile + ']';
        return GenreMap.genreMap;
