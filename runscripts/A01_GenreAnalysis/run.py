# Run and get some statistics. 
# Created: Jiayu Zhou, Jan 20, 2014. 

from DailyLog import DailyLog;
from DailyLog import GenreMap;
import csv;
import time;
from os import listdir;
from jiayuUtils import *;


genreMap = GenreMap.GetMap();

log1 = DailyLog.createFromFile('./data/20131122.tsv');

# display genre dictionary. 
for key,value in log1.genreDic.items():
    print "["+key+"]", genreMap[key], ":", len(value);

print 'Number of total entries:  ' + str(log1.length());
print 'Number of entries unique: ' + str(len(log1.progDic));

dataDir = './data';
logfiles = sorted(listdir(dataDir));


## compute the overlap week by week.
for widx in range(0, len(logfiles)/7 - 1):
    print '---Weekly----'

    if ('logArr2' in locals()):
        logArr1 = logArr2;
    else:
        logArr1 = DailyLog.mergeLogs( [DailyLog.createFromFile(dataDir + '/' + logfiles[fidx]) \
              for fidx in range(widx * 7 + 0,  widx * 7 + 7)]);

    logArr2 = DailyLog.mergeLogs( [DailyLog.createFromFile(dataDir + '/' + logfiles[fidx]) \
              for fidx in range(widx * 7 + 7,  widx * 7 + 14)]);
        
    print logfiles[widx * 7 + 0],':', logfiles[widx * 7 +  7], '-', str(len(logArr1.progDic))
    print logfiles[widx * 7 + 7],':', logfiles[widx * 7 + 14], '-', str(len(logArr2.progDic))
    
    #print 'Overlap: ', logArr1.overlap(logArr2);
    logArr1.showOverlapRatio(logArr2);
    

# compute the overlap day by day. 
for fidx in range(0, len(logfiles) - 1):
    print '----Daily----'
    filestr1 = dataDir + '/' + logfiles[fidx];
    filestr2 = dataDir + '/' + logfiles[fidx + 1];

    if ('log2' in locals()):
        log1 = log2;
    else:
        log1 = DailyLog.createFromFile(filestr1);
    log2 = DailyLog.createFromFile(filestr2);

    print logfiles[fidx],   ':', str(len(log1.progDic))
    print logfiles[fidx+1], ':', str(len(log2.progDic))

    log1.showOverlapRatio(log2);
    
    #t = time.time()
    #print 'Overlap: ', log1.overlap(log2);
    #print time.time() - t
    
    #t = time.time()

    #log12 = log1.mergeLog(log2);
    #print 'check unique program'
    #print sum(len(x) for x in log1.progDic.values())
    #print sum(len(x) for x in log2.progDic.values())
    #print sum(len(x) for x in log12.progDic.values())

    #print 'check genre'
    #print len(uniqueDicValues(log1.genreDic));
    #print len(uniqueDicValues(log2.genreDic));
    #print len(uniqueDicValues(log12.genreDic));
    # NOTE: total entries in genre may be less than those in prog, 
    #       because some entries do not have genre.

    #print 'Overlap (Merge) : ', str(len(log1.progDic) + len(log2.progDic) - len(log1.mergeLog(log2).progDic))
    #print time.time() - t
    
