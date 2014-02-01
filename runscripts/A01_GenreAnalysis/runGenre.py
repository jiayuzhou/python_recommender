# Analyze genre. 

from DailyLog import DailyLog;
from DailyLog import GenreMap;
from os import listdir;
import operator;


dataDir = './data';
logfiles = sorted(listdir(dataDir));


def genreMergeValue(genreValue1, genreValue2):
    # a value indicating if the two genre should be merged 
    unionSet = len(set(genreValue1 + genreValue2));
    interSet = len(genreValue1) + len(genreValue2) - unionSet;
    #[unionSet.append(obj) for obj in (genreValue1 + genreValue2) if obj not in unionSet];

    nominator = min( len(genreValue1), len( genreValue2 ));
    return float(interSet)/ float(nominator);
    #return float(interSet)/float(unionSet)

genreMap = GenreMap.GetMap();


# weekly aggregation. 


print 'Merging weekly data...'
widx = 0;
logArr1 = DailyLog.mergeLogs( [DailyLog.createFromFile(dataDir + '/' + logfiles[fidx]) \
              for fidx in range(widx * 7 + 0,  widx * 7 + 7)]);

#logArr1 = DailyLog.createFromFile('./data/20131122.tsv');

              
print 'Computing distance...'
kvPairs = logArr1.genreDic.items();
pairwiseComp = dict([ ( (kvPairs[i][0], kvPairs[j][0]), \
                        genreMergeValue (kvPairs[i][1], kvPairs[j][1])) \
  for i in range(0, len(kvPairs)) for j in range(0, len(kvPairs)) if i<j ]);

#print pairwiseComp

print 'Rank and output results...'
sortComp = sorted(pairwiseComp.iteritems(), key=operator.itemgetter(1), reverse = True);
sortComp = [item for item in sortComp if item[1] > 0]; # filtering zero ones. 

print "------Genre similarity------"
for item in sortComp:
    idx1 = item[0][0];
    idx2 = item[0][1];
    idx1Len = len(logArr1.genreDic[idx1]);
    idx2Len = len(logArr1.genreDic[idx2]);
    if idx1Len >= idx2Len:
        print "["+idx1+":"+genreMap[idx1]+ "(" + str(idx1Len) + ")]" + \
              "["+idx2+":"+genreMap[idx2]+ "(" + str(idx2Len) + ")]" + \
              "=>" + str(item[1]); 
    else:
        print "["+idx2+":"+genreMap[idx2]+ "(" + str(idx2Len) + ")]" + \
              "["+idx1+":"+genreMap[idx1]+ "(" + str(idx1Len) + ")]" + \
              "=>" + str(item[1]);
print "----------------------------"
