% watch time. 

clear; close all;

%%
view_status_file = 'view_watchtime.mat';



if ~exist(view_status_file, 'file')
    % construct the sparse data matrix. 
    daily_data_mat_file = '/Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131209_sparse.mat';
    load_data = load(daily_data_mat_file);

    data_mat = sparse(double(load_data.i') + 1, double(load_data.j') + 1, double(load_data.data'), ...
        double(load_data.m), double(load_data.n));
    
    user_viewtime      = full(sum(data_mat,2));
    prog_viewtime      = full(sum(data_mat,1));
    
    save(view_status_file, 'user_viewtime', 'prog_viewtime');
else
    loadstat = load(view_status_file);
    user_viewtime      = loadstat.user_viewtime;
    prog_viewtime      = loadstat.prog_viewtime;
end

user_num = length(user_viewtime);
prog_num = length(prog_viewtime);



%% plot [user percentage] versus [min number of views]
user_viewtime = user_viewtime/60/60;% hours;
max_user_time = max(user_viewtime);

x = 0:max_user_time/50:max_user_time;
y = zeros(size(x));
for i = 1: length(x)
    y(i) = sum(user_viewtime > x(i))/user_num; 
end

f3 = figure;
semilogy(x, y, 'r-*');
grid on;
xlabel('Total minutes of view');
ylabel('Fraction of users');
title('Cumulative distribution of user activities')

set(0,'DefaultAxesFontSize', 16)
set(0,'DefaultTextFontSize', 16)

%print(f3, '-dpdf', 'cum_dist_wt_user_act');

%% plot [prog percetange] versus [min number of views]
max_prog_time = max(prog_viewtime);

x = 1:max_prog_time/50:max_prog_time;
y = zeros(size(x));
for i = 1: length(x)
    y(i) = sum(prog_viewtime > x(i))/prog_num;
end

f4 = figure;
semilogy(x, y, 'r-*');
grid on;
xlabel('Total minutes of view');
ylabel('Fraction of programs');
title('Cumulative distribution of program views')

set(0,'DefaultAxesFontSize', 16)
set(0,'DefaultTextFontSize', 16)

%print(f4, '-dpdf', 'cum_dist_wt_prog_view');