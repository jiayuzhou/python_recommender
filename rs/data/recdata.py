'''
Created on Jan 28, 2014

@author: jiayu.zhou
'''

from rs.data.generic_data import GenericData;
from rs.utils.sparse_matrix import normalize_row;
from scipy.sparse import coo_matrix, csr_matrix;
import numpy as np;
import random #@UnusedImport
import rs.data.data_split as ds;
from copy import deepcopy



class FeedbackData(GenericData):
    '''
    A data structure for typical recommender system.  
    '''
    
    METAKEY_ROW_FEATURE = 'row_feature';
    METAKEY_COL_FEATURE = 'col_feature';
    
    def __init__(self, data_row, data_col, data_val, num_row, num_col,\
                  row_mapping = None, col_mapping = None, meta_data = None):
        '''
        data tuples: (data_row[i], data_col[i], data_val[i])
        number of row/col: num_row, num_col;
        name of rows/cols in a dictionary format: row_mapping, col_mapping
        '''
        # check the consistency of the data elements. 
        if (not len(data_row) == len(data_col)) or (not len(data_row) == len(data_val)):
            print;
        
        # the following things form a tuple (data_row[i], data_col[i], data_val[i]).
        # later on a data_row can construct a coo sparse matrix.  
        self.data_row = data_row[:]; 
        self.data_col = data_col[:];
        self.data_val = data_val[:];
        self.num_row  = num_row;
        self.num_col  = num_col;
        
        self.row_mapping = row_mapping.copy();
        self.col_mapping = col_mapping.copy();
        if meta_data: 
            self.meta        = meta_data.copy();
        else:
            self.meta        = [];        
    
    def __str__(self):
        if self.meta:
            return 'Row#:' + str(self.num_row) + ' Col#:' + str(self.num_col) + ' Element:'+ str(len(self.data_val)) + ' Meta available.';
        else:
            return 'Row#:' + str(self.num_row) + ' Col#:' + str(self.num_col) + ' Element:'+ str(len(self.data_val));
     
    def attach_col_feature(self, col_feature_map):
        '''
        This process attaches column-wise features. 
        
        @param col_feature_map: a <key,value> mapping where key corresponds to the 
                keys in the self.col_mapping, and the value is feature vector. Could be  
                any object. 
                
                For format of the feature object refer to the algorithms that utilizes
                the information.
                
                This builds col_feature stored in self.meta[METAKEY_COL_FEATURE]
                col_feature: kk, vv => kk: idx, vv: feature
        '''
        
        if not self.meta:
            self.meta = {};
        
        # the following list comprehension performs the following. 
        # for each kk,vv in col_mapping (kk: pid, vv: id)
        #    if kk in col_feature_map
        #        col_feature[vv] = col_feature_map[kk]
        #    else
        #        col_feature[vv] = []
        col_feature = dict([ (vv, col_feature_map[kk]) if kk in col_feature_map \
                        else (vv, []) for kk,vv in self.col_mapping.iteritems() ])  
        
        self.meta[FeedbackData.METAKEY_COL_FEATURE] = col_feature;
     
    def get_sparse_matrix(self):
        '''
        Construct a sparse matrix (coo_matrix) from current Feedback data content.
        
        TODO: cache the data matrix. 
        --------
        @return: the COO sparse matrix representation of the current data.   
        '''
        mat = coo_matrix((self.data_val, (self.data_row, self.data_col)), \
                     shape = (self.num_row, self.num_col));
        return mat;
    
    def get_cold_start_col(self):
        '''
        Get a list of indices of columns that have no values, i.e., the cold start columns.
        
        Returns
        ---------
        @return: a list of 0-based column indices.
        '''
        # get a column compressed matrix cc, then we have access to cc.indptr.  
        cc = self.get_sparse_matrix().tocsc();
        # if a column has no elements then in cc.indptr[col] = cc.indptr[col + 1];
        cs_col = [i for i in range(cc.shape[1]) if cc.indptr[i] == cc.indptr[i+1]];
        return cs_col;
    
    def get_cold_start_row(self):
        '''
        Get a list of indices of rows that have no values, i.e., the cold start rows.
        
        Returns
        ---------
        @return: a list of 0-based row indices.
        '''
        # get a row compressed matrix rr, then we have access to rr.indptr.  
        rr = self.get_sparse_matrix().tocsr();
        # if a row has no elements then in rr.indptr[row] = rr.indptr[row + 1];
        cs_row = [i for i in range(rr.shape[0]) if rr.indptr[i] == rr.indptr[i+1]];
        return cs_row;
    
    def binarize(self):
        '''
        Change all non-zero values (data) to one. 
        '''
        self.data_val = [1] * len(self.data_val); 
    
    def normalize_row(self):
        '''
        Perform a normalization on the row. This will modify the row/col/val.
        The method first constructs a coo sparse matrix and then convert to lil 
        matrix for normalization. The normalized matrix is then converted back to 
        coo matrix and row/col/val are reset to the values in the converted coo matrix. 
        
        NOTE: when we have (row, col, 0.1) and (row, col, 0.2), and we will have 
              (row, col, 0.3), because coo to lil transformation.  
        '''
        mat = self.get_sparse_matrix();
        mat = normalize_row(mat); # normalize each row. 
        mat = mat.tocoo();
        
        self.data_val = mat.data.tolist();
        self.data_row = mat.row.tolist();
        self.data_col = mat.col.tolist();
    
    def blind_k_out(self, blind_k_out_idx):
        '''
        Construct blind k out dataset (and remaining).
        
        Parameters
        ----------
        @param blind_k_out_idx
        
        Return
        @return out
        '''
        mat = self.get_sparse_matrix();
        mat_blind = mat.tolil();
        mat_rest  = mat.tolil();
        
        mat_blind[:, blind_k_out_idx] = 0;
        mat_rest [:, list(set(range(self.num_col)) - set(blind_k_out_idx))] = 0;
        
        mat_blind = mat_blind.tocoo();
        mat_rest  = mat_rest.tocoo();
        
        #blinded for training
        blind_fbdata = FeedbackData(mat_blind.row.tolist(), mat_blind.col.tolist(), mat_blind.data.tolist(),\
                                  self.num_row, self.num_col, self.row_mapping, self.col_mapping, self.meta);
        
        #the rest of data for testing. 
        rest_fbdata = FeedbackData(mat_rest.row.tolist(), mat_rest.col.tolist(), mat_rest.data.tolist(),\
                                  self.num_row, self.num_col, self.row_mapping, self.col_mapping, self.meta);
                                  
        return [blind_fbdata, rest_fbdata];
    
    def leave_k_out(self, leave_k_out_idx):
        '''
        Construct leave-k-out dataset (and remaining). 
        See test_recdata_leavekout.py for example. 
        
        Parameters
        ----------
        @param leave_k_out_idx
        
        Returns
        ----------
        @return out
        '''
        
        # get the sparse matrix. 
        smat = self.get_sparse_matrix().tocsr();
        
        # matrix size: self.num_col; self.num_row;
        
        tr_data    = [];
        tr_indices = [];
        tr_indptr  = [];
        
        lo_data    = [];
        lo_indices = [];
        lo_indptr  = [];
        
        # 
        tr_idx = 0;
        lo_idx = 0;
        
        tr_indptr.append(tr_idx);
        lo_indptr.append(lo_idx);
        
        for row_pos in range(self.num_row):
            row_leave_idx = leave_k_out_idx[row_pos];
            
            # col_pos_idx is in range [range(smat.indptr[row_pos], smat.indptr[row_pos + 1]]
            # and the indices in 
            # Access data at col_pos_idx.   smat.data    [col_pos_idx];
            # Access column index:          smat.indices [col_pos_idx];
            for col_pos_idx in range(smat.indptr[row_pos], smat.indptr[row_pos + 1]):
                
                col_pos = smat.indices[col_pos_idx]; # the column position. 
                
                if col_pos in row_leave_idx:
                    lo_data.append   (smat.data[col_pos_idx]);
                    lo_indices.append(col_pos);
                    lo_idx += 1;
                else:
                    tr_data.append   (smat.data[col_pos_idx]);
                    tr_indices.append(col_pos);
                    tr_idx += 1;
                
            lo_indptr.append(lo_idx);
            tr_indptr.append(tr_idx);
            # end for each column.
        # end for each row. 
        
        lo_data    = np.array(lo_data);
        lo_indices = np.array(lo_indices);
        lo_indptr  = np.array(lo_indptr);
        
        tr_data    = np.array(tr_data);
        tr_indices = np.array(tr_indices);
        tr_indptr  = np.array(tr_indptr);
        
        # construct left out feedback dataset. 
        lo_data   = csr_matrix( (lo_data,lo_indices,lo_indptr), shape=(self.num_row, self.num_col)).tocoo()
        lo_fbdata = FeedbackData(lo_data.row.tolist(), lo_data.col.tolist(), lo_data.data.tolist(),\
                                  self.num_row, self.num_col, self.row_mapping, self.col_mapping, self.meta);  
        
        # construct remaining feedback dataset.
        tr_data   = csr_matrix( (tr_data,tr_indices,tr_indptr), shape=(self.num_row, self.num_col)).tocoo()
        tr_fbdata = FeedbackData(tr_data.row.tolist(), tr_data.col.tolist(), tr_data.data.tolist(),\
                                  self.num_row, self.num_col, self.row_mapping, self.col_mapping, self.meta);
        
        return [lo_fbdata, tr_fbdata];
        
    
    def subdata_row(self, sel_idx):
        '''
        select a set of rows (users) to form a new dataset.  
        
        Parameters
        ----------
        @param sel_idx: a list of row indices. The row index set can have duplicated entries.  
        
        Returns
        ----------
        @return out: a new FeedbackData instance whose rows are given in sel_idx 
             in the original set. 
        '''
        # check if the indices are good. 
        if max(sel_idx) >= self.num_row or min(sel_idx) < 0:
            raise ValueError('Found invalid element in the index set!');
        
        # construct sparse matrix using coo.
        mat = self.get_sparse_matrix();
                     
        # convert to csr for fast row slicing. 
        mat = mat.tocsr();
        
        # sub-sampling data matrix.
        mat = mat[sel_idx, :];            # slice the matrix. 
        mat = mat.tocoo();               # convert it back. 
        
        # recompute the row mapping. 
        inv_mapping = {yy : xx for xx, yy in self.row_mapping.iteritems()};
        row_mapping = { inv_mapping[xx]: ii for ii, xx in enumerate(sel_idx) };
        
        data_val = mat.data.tolist();
        data_row = mat.row.tolist();
        data_col = mat.col.tolist();
        
        
        newdata = FeedbackData(data_row, data_col, data_val, len(sel_idx), self.num_col,\
                  row_mapping, self.col_mapping, self.meta); # generate new data 
              
        return newdata;
    
    def subdata_col(self, sel_idx):
        '''
        Select a set of columns (programs) to form a new dataset.
        '''
        # check if the indices are good. 
        if max(sel_idx) >= self.num_col or min(sel_idx) < 0:
            raise ValueError('Found invalid element in the index set!');
        
        # construct sparse matrix using coo.
        mat = self.get_sparse_matrix();
        
        # convert to csc for fast column slicing. 
        mat = mat.tocsc();
        
        # sub-sampling data.
        mat = mat[:, sel_idx];  # slice the matrix.
        mat = mat.tocoo();      # convert it back. 
        
        # recompute the row mapping.
        inv_mapping = {yy: xx for xx, yy in self.col_mapping.iteritems()};
        col_mapping = {inv_mapping[xx]: ii for ii, xx in enumerate(sel_idx)}; 
        
        data_val = mat.data.tolist();
        data_row = mat.row.tolist();
        data_col = mat.col.tolist();
        
        newdata = FeedbackData(data_row, data_col, data_val, self.num_row, len(sel_idx), \
                        self.row_mapping, col_mapping, self.meta); # generate new data.
        
        return newdata;
        
    def subsample_row(self, sel_row_num):
        '''
        randomly sub-sample the data of a given number of users. This will modify 
        the row/col/val. 
        The method first constructs a coo sparse matrix and then convert to csr matrix 
        for row slicing. And then the csr matrix is converted back.
        
        Parameters
        ----------
        @param sel_row_num: the number of selected rows. 
        
        Returns
        ----------
        @return out: a list of two components [sample_data, selidx]
        sample_data: 
        sel_idx:
        '''
        
        # sampling a set of rows 
        sel_idx = ds.sample_num(self.num_row, sel_row_num);
        
        # construct data set using the selected rows 
        sample_data = self.subdata_row(sel_idx);
        return [sample_data, sel_idx];
        
    def split(self, percentage):
        '''
        get a random splitting of data with a specified proportion of rows. 
        NOTE: it is recommended to use subdata_row method in get deterministic splits. 
        
        Parameters
        ----------
        @param percentage: the percentage of data split. 
        
        Returns
        ----------
        @return out: a list [data_split, data_split_comp, selidx_split, selidx_split_comp]
        data_split: a Feedback data of percentage, whose index (in the full data set) 
                    is given in selidx_split
        data_split_comp: a Feedback data of 1- percentage, whose index is given 
                         in data_split_comp. This is the complement part of data_split. 
        selidx_split: the index of rows in data_split.
        selidx_split_comp: the index of rows in data_split_comp. 
        '''
        
        # obtain the indices of the split / complement of the split.  
        [selidx_split, selidx_split_comp] = ds.split(self.num_row, percentage);
        
        # acquire data from the splits. 
        data_split      = self.subdata_row(selidx_split);
        data_split_comp = self.subdata_row(selidx_split_comp);
        
        return [data_split, data_split_comp, selidx_split, selidx_split_comp];
    
    def fold(self, fold_num, total_fold):
        '''
        get the n-th fold of the n-fold data.
        
        Parameters
        ----------
        @param fold_num:    the 0-based fold index [0..total_fold-1] 
        @param total_fold:  the total number the (n) of n-fold. 
        
        Returns
        ----------
        @return out: [fold_data, sel_idx]
        fold_data: Feedback data structure. 
        sel_idx: fold index
        '''
        
        sel_idx = ds.fold(self.num_row, fold_num, total_fold);
        fold_data = self.subdata_row(sel_idx);
        
        return [fold_data, sel_idx];




def merge_data(fb_data1, fb_data2, fb_data1_col_iden = None, fb_data2_col_iden = None):
    '''
    Merge two Feedback data structures. 
    
    An optional operation before merge is to add identifications to the column mappings keywords  
    '''
    
    # if identifier is defined, add identifiers to the mapping keywords. 
    if fb_data1_col_iden:
        fb_data1 = deepcopy(fb_data1)
        fb_data1.col_mapping = dict([ (fb_data1_col_iden + ':'+kk, vv) \
                                     for kk, vv in fb_data1.col_mapping.iteritems() ])
    if fb_data2_col_iden:
        fb_data2 = deepcopy(fb_data2)
        fb_data2.col_mapping = dict([ (fb_data2_col_iden + ':'+kk, vv) \
                                     for kk, vv in fb_data2.col_mapping.iteritems() ])
    
    # prepare to merge.
    merge_row  = []
    merge_col  = []
    merge_data = []
    
    merge_row_keyset = set(fb_data1.row_mapping.keys()).union(fb_data2.row_mapping.keys())
    merge_col_keyset = set(fb_data1.col_mapping.keys()).union(fb_data2.col_mapping.keys())

    merge_row_mapping = dict([ (kk, vv) for kk, vv in enumerate(sorted(merge_row_keyset)) ])
    merge_col_mapping = dict([ (kk, vv) for kk, vv in enumerate(sorted(merge_col_keyset)) ])
    
    fb_data1_col_remap
    fb_data1_row_remap
    fb_data2_col_remap
    fb_data2_row_remap
    
    
    print merge_row_mapping
    print merge_col_mapping

def share_row_data(fb_data1, fb_data2):
    '''
    Compute the shared data. The [row_mapping]s of the Feedback data 
    are used.
    
    This method firstly find the shared rows and then construct two corresponding 
    Feedback datasets, each of which only includes the shared rows. The rows of 
    the two datasets are aligned.   
    
    Parameters
    ----------
    @param fb_data1:
    @param fb_data2:
    
    Returns
    ----------
    @return out: [fb_data1_share, fb_data2_share]
    fb_data1_share: 
    fb_data2_share: 
    '''
    
    if (not fb_data1.row_mapping) or (not fb_data2.row_mapping):
        raise ValueError('Both input Feedback  should ');
    
    # obtain the list of shared users. 
    row_mapping1 = fb_data1.row_mapping;
    row_mapping2 = fb_data2.row_mapping;
    shared_user  = []; 
    if len(row_mapping1) < len(row_mapping2): 
        # for rows in row_mapping1
        for row_id in row_mapping1.keys():
            if row_id in row_mapping2:
                shared_user.append(row_id);
    else:
        # for rows in row_mapping2
        for row_id in row_mapping2.keys():
            if row_id in row_mapping1:
                shared_user.append(row_id);
    
    # build selection indices. 
    sel_idx1 = [];
    for row_id in shared_user:
        sel_idx1.append(row_mapping1[row_id]);
    
    sel_idx2 = [];
    for row_id in shared_user:
        sel_idx2.append(row_mapping2[row_id]);
    
    # construct sub-data. 
    fb_data1_share = fb_data1.subdata_row(sel_idx1);
    fb_data2_share = fb_data2.subdata_row(sel_idx2);
    return [fb_data1_share, fb_data2_share];


