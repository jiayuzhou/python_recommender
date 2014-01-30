'''
This program computes and outputs the similar programs.
The results are prepared for Wook's team. 

Deadline is Jan 31st, 2014.

Created on Jan 29, 2014

@author: jiayu.zhou
'''

import numpy as np;
import csv;
import cPickle as pickle;
import os; #@UnusedImport
import sys;

from rs.utils.log import Logger;
from scipy.sparse import coo_matrix;
from rs.data.daily_watchtime import DailyWatchTimeReader
from rs.utils.sparse_matrix import normalize_row;
from scipy.spatial.distance import cosine;

log = lambda message: Logger.Log('PROG SIMILARITY: '+message, Logger.MSG_CATEGORY_EXP);

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        filename        = sys.argv[1];
        rovi_daily_file = sys.argv[2];
    else:
        # INPUT: the ROVI daily mapping. 
        #  Hadoop location: /apps/vddil/rovi_daily
        rovi_daily_file = "/Users/jiayu.zhou/Data/rovi_daily/20131209.tsv";
        
        # INPUT: the aggregated data file. 
        #  Hadoop location: /apps/vddil/duid-program-watchTime-genre
        #filename = "/Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131209/part-r-00000";
        filename = "../../datasample/agg_duid_pid_watchtime_genre/20131209_100000";
    
    log('Data file: ' + filename);
    log('ROVI daily file: ' + rovi_daily_file);    
    
    # build ROVI daily mapping    
    log('Building ROVI daily mapping');
    rovi_mapping = {};
    with open(rovi_daily_file) as csvfile:
        rovi_reader = csv.reader(csvfile, delimiter = '\t', quotechar = '|')
        for row in rovi_reader:
            rovi_mapping[row[3]] = row[1];
    
    # load data from file and transform into a sparse matrix. 
    reader = DailyWatchTimeReader();
    fbdata = reader.read_file_with_minval(filename, 5, 5);  
    
    mat = coo_matrix((fbdata.data_val, (fbdata.data_row, fbdata.data_col)), \
                     shape = (fbdata.num_row, fbdata.num_col));
    # memo: if we do multiple days, we might use coo_matrix summation, but we need
    #       to align the program and user.    
    
    # we have a mapping from program id to row.
    program_mapping = fbdata.col_mapping; 
    # from which we build a reverse mapping from row id to program
    # the reverse mapping allows us to find program ID from matrix position.
    program_inv_mapping = {y: x for x, y in program_mapping.items()};
    # check the consistency. 
    if not (len(program_mapping) == len(program_inv_mapping)):
        raise ValueError('Mapping inverse error!');
    program_num = len(program_mapping);
    
    # compute pairwise similarity
    similarity_file = filename + '.prgsim';
    if not os.path.isfile(similarity_file):
        # normalize data per user. 
        log('normalizing data...')
        mat = normalize_row(mat);
    
        log('computing pairwise similarity...');
        total_pair = program_num *  (program_num + 1) / 2;
        progress = 0;
        
        cor_mat = np.zeros((program_num, program_num));
        for i in range(program_num):
            for j in range(program_num):
                if i < j:
                    progress += 1;
                    if progress % 1000 == 0:
                        log('Computing '+ str(progress) + ' out of ' + str(total_pair));
                    
                    # Our similarity is defined as [1 - cosine distance]. 
                    cor_mat[i][j] = 1 - cosine(mat.getcol(i).todense(), mat.getcol(j).todense());
                    cor_mat[j][i] = cor_mat[i][j];
        
        log('saving the similarity to file...')
        pickle.dump(cor_mat, open(similarity_file,"wb"));
    else:
        log('computed similarity found. loading...')
        cor_mat = pickle.load(open(similarity_file, "rb"));
        
    # output list. 
    for i in range(program_num):
        # take each column and rank the list.  
        val = (cor_mat[:, i]).tolist();
        
        # sort array with index.
        # http://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
        srt_list = [(k[0], k[1]) for k in sorted(enumerate(val), key=lambda x:x[1], reverse=True)];
        
        # program name.
        print '----------------'; 
        print 'Program: ' + rovi_mapping[program_inv_mapping[i]] + ' [' + program_inv_mapping[i] + ']';
        print 'Top Similar Programs:' 
        # most similar programs . 
        for j in range(20):
            if srt_list[j][1]>0:
                iter_prog_id = program_inv_mapping[srt_list[j][0]];
                if iter_prog_id in rovi_mapping:
                    #print srt_list[j][0], iter_prog_id, rovi_mapping[iter_prog_id], srt_list[j][1];
                    print rovi_mapping[iter_prog_id] +'[' + iter_prog_id + '] \t', srt_list[j][1];
                else:
                    #print srt_list[j][0], iter_prog_id, 'UNKNOWN PRG', srt_list[j][1];
                    print 'UNKNOWN PRG [' + iter_prog_id + '] \t' + srt_list[j][1];
    print 'done';




