%
% Plot demographic information. 
%
clear; close all;


%% load data. 
view_status_file = 'view_status.mat';

if ~exist(view_status_file, 'file')
    daily_data_mat_file = '/Users/jiayu.zhou/Data/duid-program-watchTime-genre/20131209.mat';
    loaddata = load(daily_data_mat_file);

    user_view_prog     = full(sum(loaddata.data>0,2));
    prog_viewedby_user = full(sum(loaddata.data>0,1));
    
    user_viewtime      = full(sum(loaddata.data,2));
    prog_viewtime      = full(sum(loaddata.data,1));
    
    save(view_status_file, 'user_view_prog', 'prog_viewedby_user', ...
        'user_viewtime', 'prog_viewtime');
else
    loadstat = load(view_status_file);
    user_view_prog     = loadstat.user_view_prog;
    prog_viewedby_user = loadstat.prog_viewedby_user;
    user_viewtime      = loadstat.user_viewtime;
    prog_viewtime      = loadstat.prog_viewtime;
end

user_num = length(user_view_prog);
prog_num = length(prog_viewedby_user);

%% plot [user percentage] versus [min number of views]
x = 1:2:500;
y = zeros(size(x));
for i = 1: length(x)
    y(i) = sum(user_view_prog > x(i))/user_num; 
end

f1 = figure;
%semilogy(x, y, '-*');
loglog(x, y, '-*');
grid on;
xlabel('Minimum number of views');
ylabel('Fraction of users');
title('Cumulative distribution of user activities')

set(0,'DefaultAxesFontSize', 16)
set(0,'DefaultTextFontSize', 16)

print(f1, '-dpdf', 'cum_dist_user_act');

%% plot [prog percetange] versus [min number of views]
x2 = 1:100:10000;
y2 = zeros(size(x2));
for i = 1: length(x2)
    y2(i) = sum(prog_viewedby_user > x2(i))/prog_num;
end

f2 = figure;
%semilogy(x, y, '-*');
loglog(x2, y2, '-*');
grid on;
xlabel('Minimum number of views');
ylabel('Fraction of programs');
title('Cumulative distribution of program views')

set(0,'DefaultAxesFontSize', 16)
set(0,'DefaultTextFontSize', 16)

print(f2, '-dpdf', 'cum_dist_prog_view');


% %% plot [user percentage] versus [min number of views]
% max_user_time = max(user_viewtime);
% 
% x = 0:user_viewtime/50:max_user_time;
% y = zeros(size(x));
% for i = 1: length(x)
%     y(i) = sum(user_viewtime > x(i))/user_num; 
% end
%
% f3 = figure;
% semilogy(x, y, 'r-*');
% grid on;
% xlabel('Total minutes of view');
% ylabel('Fraction of users');
% title('Cumulative distribution of user activities')
% 
% set(0,'DefaultAxesFontSize', 16)
% set(0,'DefaultTextFontSize', 16)
% 
% %print(f3, '-dpdf', 'cum_dist_wt_user_act');
% 
% %% plot [prog percetange] versus [min number of views]
% max_prog_time = max(prog_viewtime);
% 
% x = 1:max_prog_time/50:max_prog_time;
% y = zeros(size(x));
% for i = 1: length(x)
%     y(i) = sum(prog_viewtime > x(i))/prog_num;
% end
% 
% f4 = figure;
% semilogy(x, y, 'r-*');
% grid on;
% xlabel('Total minutes of view');
% ylabel('Fraction of programs');
% title('Cumulative distribution of program views')
% 
% set(0,'DefaultAxesFontSize', 16)
% set(0,'DefaultTextFontSize', 16)
% 
% %print(f4, '-dpdf', 'cum_dist_wt_prog_view');