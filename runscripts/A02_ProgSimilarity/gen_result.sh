#!/bin/bash 

# generate results one by one.

echo 20131209 Top 50 with duplication...
PYTHONPATH=~/workspace/recsys/ python ./gen_program_similarity.py /Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131209/part-r-00000 /Users/jiayu.zhou/Data/rovi_daily/20131209.tsv 50 True > top50dup_20131209.tsv 

echo 20131210 Top 50 with duplication...
PYTHONPATH=~/workspace/recsys/ python ./gen_program_similarity.py /Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131210/part-r-00000 /Users/jiayu.zhou/Data/rovi_daily/20131210.tsv 50 True > top50dup_20131210.tsv 

echo 20131209 Top 20 without duplication...
PYTHONPATH=~/workspace/recsys/ python ./gen_program_similarity.py /Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131209/part-r-00000 /Users/jiayu.zhou/Data/rovi_daily/20131209.tsv 20 False > top20_20131209.tsv

echo 20131210 Top 20 without duplication...
PYTHONPATH=~/workspace/recsys/ python ./gen_program_similarity.py /Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131210/part-r-00000 /Users/jiayu.zhou/Data/rovi_daily/20131210.tsv 20 False > top20_20131210.tsv 
