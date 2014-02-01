
from jiayuUtils import *;

dicts0 = {'a':1, 'b':2, 'c':3}
dicts1 = {'a':1, 'd':2, 'c':'foo'}

#print combineTwoDictList(dicts0, dicts1);

dicts = {};
dicts[0] = {'a':[1], 'b':[2], 'c':[3]}
dicts[1] = {'a':[1], 'd':[2], 'c':['foo']}
dicts[2] = {'e':[57],'c':[3]}

dicts = [{'a':[1], 'b':[2], 'c':[3]},
         {'a':[1], 'd':[2], 'c':['foo']},
         {'e':[57], 'c':[3, 5]} ]

print combineSetDictList(dicts);

print combineSetDictList(dicts, False);

dict2 = {'a':[1,2], 'b':[1, 3], 'c':[5]};

print uniqueDicValues(dict2);
