% This script list of training and testing data files which will be
% processed by SeqSleepNet, and two baselines E2E-ARNN and Multitask E2E-ARNN in tensorflow (for efficiency)

clear all
close all
clc


%% Generate splits from csv files

T = [importfile('./../data/csv/isruc.csv');
     importfile('./../data/csv/mros.csv');
     importfile('./../data/csv/shhs.csv');
     importfile('./../data/csv/ssc.csv');
     importfile('./../data/csv/wsc.csv')];
T(find(~cellfun(@isempty, T.Skip)), :) = [];
N_train = sum(cellfun(@(x) strcmpi(x, 'train'), T.Partition));
N_eval = sum(cellfun(@(x) strcmpi(x, 'eval'), T.Partition));
N_test = sum(cellfun(@(x) strcmpi(x, 'test'), T.Partition));
train_sub = cell(1, 1);
eval_sub = cell(1, 1);
test_sub = cell(1, 1);
train_sub{1, 1} = find(cellfun(@(x) strcmpi(x, 'train'), T.Partition))';
eval_sub{1, 1} = find(cellfun(@(x) strcmpi(x, 'eval'), T.Partition))';
test_sub{1, 1} = find(cellfun(@(x) strcmpi(x, 'test'), T.Partition))';

%%

% load('./data_split_eval.mat');

mat_path = './../mat/';
Nfold = 1;

%% EEG
listing = dir([mat_path, '*_seqsleepnet_eeg.mat']);
tf_path = './../tf_data/seqsleepnet_eval_eeg/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

for s = 1 : Nfold

    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
    
	train_s = train_sub{s};
    eval_s = eval_sub{s};
    test_s = test_sub{s};
    
    train_filename = [tf_path, 'train_list_n', num2str(s),'.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
%         if ~exist([mat_path, lower(T.FileID{train_s(i)}), '_seqsleepnet_eeg.mat'], 'file')
%             continue;
%         end
        sname = [lower(T.FileID{train_s(i)}), '_seqsleepnet_eeg.mat'];
%         sname = listing(train_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['./../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list_n', num2str(s),'.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
%         if ~exist([mat_path, lower(T.FileID{test_s(i)}), '_seqsleepnet_eeg.mat'], 'file')
%             continue;
%         end
        sname = [lower(T.FileID{test_s(i)}), '_seqsleepnet_eeg.mat'];
%         sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['./../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    eval_filename = [tf_path, 'eval_list_n', num2str(s),'.txt'];
    fid = fopen(eval_filename,'wt');
    for i = 1 : numel(eval_s)
%         if ~exist([mat_path, lower(T.FileID{eval_s(i)}), '_seqsleepnet_eeg.mat'], 'file')
%             continue;
%         end
        sname = [lower(T.FileID{eval_s(i)}), '_seqsleepnet_eeg.mat'];
%         sname = listing(eval_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['./../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
end


%% EOG
listing = dir([mat_path, '*_seqsleepnet_eog.mat']);
tf_path = './../tf_data/seqsleepnet_eval_eog/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

for s = 1 : Nfold

    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
    
	train_s = train_sub{s};
    eval_s = eval_sub{s};
    test_s = test_sub{s};
    
    train_filename = [tf_path, 'train_list_n', num2str(s),'.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
%         if ~exist([mat_path, lower(T.FileID{train_s(i)}), '_seqsleepnet_eog.mat'], 'file')
%             continue;
%         end
        sname = [lower(T.FileID{train_s(i)}), '_seqsleepnet_eog.mat'];
%         sname = listing(train_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['./../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list_n', num2str(s),'.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
%         if ~exist([mat_path, lower(T.FileID{test_s(i)}), '_seqsleepnet_eog.mat'], 'file')
%             continue;
%         end
        sname = [lower(T.FileID{test_s(i)}), '_seqsleepnet_eog.mat'];
%         sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['./../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    eval_filename = [tf_path, 'eval_list_n', num2str(s),'.txt'];
    fid = fopen(eval_filename,'wt');
    for i = 1 : numel(eval_s)
%         if ~exist([mat_path, lower(T.FileID{eval_s(i)}), '_seqsleepnet_eog.mat'], 'file')
%             continue;
%         end
        sname = [lower(T.FileID{eval_s(i)}), '_seqsleepnet_eog.mat'];
%         sname = listing(eval_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['./../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
end


%% EMG
listing = dir([mat_path, '*_seqsleepnet_emg.mat']);
tf_path = './../tf_data/seqsleepnet_eval_emg/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

for s = 1 : Nfold

    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
    
	train_s = train_sub{s};
    eval_s = eval_sub{s};
    test_s = test_sub{s};
    
    train_filename = [tf_path, 'train_list_n', num2str(s),'.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
%         if ~exist([mat_path, lower(T.FileID{train_s(i)}), '_seqsleepnet_eog.mat'], 'file')
%             continue;
%         end
        sname = [lower(T.FileID{train_s(i)}), '_seqsleepnet_eog.mat'];
%         sname = listing(train_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['./../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list_n', num2str(s),'.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
%         if ~exist([mat_path, lower(T.FileID{test_s(i)}), '_seqsleepnet_eog.mat'], 'file')
%             continue;
%         end
        sname = [lower(T.FileID{test_s(i)}), '_seqsleepnet_eog.mat'];
%         sname = listing(test_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['./../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
    
    eval_filename = [tf_path, 'eval_list_n', num2str(s),'.txt'];
    fid = fopen(eval_filename,'wt');
    for i = 1 : numel(eval_s)
%         if ~exist([mat_path, lower(T.FileID{eval_s(i)}), '_seqsleepnet_eog.mat'], 'file')
%             continue;
%         end
        sname = [lower(T.FileID{eval_s(i)}), '_seqsleepnet_eog.mat'];
%         sname = listing(eval_s(i)).name;
        load([mat_path,sname], 'label');
        num_sample = numel(label);
        file_path = ['./../mat/',sname];
        fprintf(fid, '%s\t%d\n', file_path, num_sample);
    end
    fclose(fid);
    clear fid file_path
end




