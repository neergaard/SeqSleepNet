function prepare_seqsleepnet_data(n, n_tot);

% clear all
% close all
% clc

% raw_data_path = './raw_data/';
raw_data_path = './../data/h5/';
mat_path = './../mat/';
if(~exist(mat_path, 'dir'))
    mkdir(mat_path);
end

fs = 100; % sampling frequency
win_size  = 2;
overlap = 1;
nfft = 2^nextpow2(win_size*fs);

% list all subjects
% listing = dir([raw_data_path, 'SS*']);
listing = dir([raw_data_path, '*.h5']);
n_listing = numel(listing);
disp(n_listing);
b = ceil(n_listing/n_tot);
disp(b);
c = mat2cell(listing, diff([0:b:n_listing-1, n_listing]));
disp(c);
listing = c{n};

for i = 1 : numel(listing)
	disp(listing(i).name)
    
%     load([raw_data_path, listing(i).name]);
    data = h5read([raw_data_path, listing(i).name], '/data');
    [M, N, C] = size(data);
    data = double(reshape(data, [], size(data, 3), 1));
    labels = h5read([raw_data_path, listing(i).name], '/hypnogram');
    [~, filename, ~] = fileparts(listing(i).name);

    % label and one-hot encoding
    label = double(labels);
    y = zeros(size(label, 1), 5);
    for k = 1 : size(y, 1)
        y(k, label(k)+1) = 1;
    end
%     y = double(labels);
%     label = zeros(size(y,1),1);
%     for k = 1 : size(y,1)
%         [~, label(k)] = find(y(k,:));
%     end
    clear labels
    
    %% Resampling to 100 Hz and diffing EOG
    % (N epochs, 30 s * 100 Hz, 3 channels)
%     print('hej')
    orig_fs = 128; % .h5 files are in 128 Hz format.
    data_ = resample(data, fs, orig_fs);
    data = zeros(size(data_, 1), 3);
    data(:, 1) = data_(:, 1);
    data(:, 2) = data_(:, 2) - data_(:, 3);
    data(:, 3) = data_(:, 4);
    data = reshape(data, [], N, 3);
    data = permute(data, [2, 1, 3]);
    clear data_
    
    %% EEG
    N = size(data, 1);
    X = zeros(N, 29, nfft/2+1);
    eeg_epochs = squeeze(data(:,:,1)); % eeg channel
    for k = 1 : size(eeg_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(size(eeg_epochs, 1))]);
        end
        [Xk,~,~] = spectrogram(eeg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X(k,:,:) = Xk;
    end
    X = single(X);
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_seqsleepnet_eeg.mat'], 'X', 'label', 'y', '-v7.3');
    clear X
    
    %% EOG
    X= zeros(N, 29, nfft/2+1);
    eeg_epochs = squeeze(data(:,:,2)); % eog channel
    for k = 1 : size(eeg_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(size(eeg_epochs, 1))]);
        end
        [Xk,~,~] = spectrogram(eeg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X(k,:,:) = Xk;
    end
    X = single(X);
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_seqsleepnet_eog.mat'], 'X', 'label', 'y', '-v7.3');
    clear X
    
    %% EMG
    X= zeros(N, 29, nfft/2+1);
    eeg_epochs = squeeze(data(:,:,3)); % emg channel
    for k = 1 : size(eeg_epochs, 1)
        if(mod(k,100) == 0)
            disp([num2str(k),'/',num2str(size(eeg_epochs, 1))]);
        end
        [Xk,~,~] = spectrogram(eeg_epochs(k,:), hamming(win_size*fs), overlap*fs, nfft);
        % log magnitude spectrum
        Xk = 20*log10(abs(Xk));
        Xk = Xk';
        X(k,:,:) = Xk;
    end
    X = single(X);
    y = single(y);
    label=single(label);
    save([mat_path, filename,'_seqsleepnet_emg.mat'], 'X', 'label', 'y', '-v7.3');
    clear X y label
end

end
