Computing Program Similarity
==============

Summary
--------------

This folder contains codes for running program similarity from user view behaviors. The similarity is based on cosine similarity. 


Data
--------------
The program requires **daily watch time (DWT) data** and **ROVI daily (RD) data** for one day, where the DWT data has the format of:

> [0]DUID \t [1]Program ID \t [2]Watchtime \t [3]Genre 

and the RD data has the format of:

> [0]anything \t [1]anything \t [2]anything \t [3]Program ID \t [4]anything \t [5]anything \t [6]anything \t [7]Program Name \t anything
 
 
Procedure
--------------
The logic procedure of the code is as follows:

* Build Program ID to Program Name mapping from RD data. 
* Load the user watch time data into a *user* by *program* sparse matrix, where the value in each cell means the watch time of a user on a specific program. 
* Normalize the matrix in a row-wise fashion, such that the sum of the normalized watch time for each user is one. After the normalization each row indicates the a user's preference. 
* Compute the pairwise similarity of each program using *1 - cosine distance*. 
* For each program, we rank all other programs according to their similarity (in the descending order), and output top *k* similar programs, where *k* is given in the input.   

How to Run
--------------
The format is:

```shell
python gen_program_similarity.py [DWT data] [RD data] k [RemoveDuplicate?]
```

where *k* is a positive integer meaning top *k* results to display, and *RemoveDuplicate* is a boolean ("True" means yes) option whether or not we remove the duplicated programs in the final list (there are programs that have multiple program IDs). 

A shell is given by gen_result.sh

