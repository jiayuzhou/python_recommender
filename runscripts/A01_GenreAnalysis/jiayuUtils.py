
from collections import defaultdict

def combineSetDictList(dicts, unique = True):
    # a list of dictionaries, each of which is (key, list)
    # this function merges the list of dictionaries. 
    # unique = True: the merge removes duplicates. 
    #          False: the merge does not remove duplicates. 
    super_dict = {};
    for d in dicts:
        for k, v in d.iteritems():
            if (k in super_dict):
                super_dict[k] = super_dict[k] + v;
            else:
                super_dict[k] = v;

    if (unique):
        for k, v in super_dict.items():
            super_dict[k] = list(set(super_dict[k]));
                
    return super_dict;

def uniqueDicValues(dictOfList):
    # combine the values in a dictionary of (key, list)
    uniqueSet = set([]);
    for tt in dictOfList.values():
        [ uniqueSet.add(t) for t in tt ]
    return uniqueSet;

