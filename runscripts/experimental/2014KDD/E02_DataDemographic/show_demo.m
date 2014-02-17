clear; close all;

daily_data_mat_file = '/Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131209.mat';
loaddata = load(daily_data_mat_file);


disp('number of programs viewed by at least 10 ppl.')
disp(nnz(full(sum(loaddata.data>0,1))>10));


disp('number of users viewd at least 10 programs')
disp(nnz(full(sum(loaddata.data>0,2))>10));

disp('number of watch events by those users on those programs')

disp(nnz(loaddata.data(full(sum(loaddata.data>0,2))>10, full(sum(loaddata.data>0,1))>10)))