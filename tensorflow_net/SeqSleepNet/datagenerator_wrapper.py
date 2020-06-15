# Sequence data generation for sequence-to-sequence sleep staging
# X: time-frequency input
# label: labels of sleep stage
# y: one-hot encoding of labels
import numpy as np
import h5py
from joblib import Parallel, Memory, delayed
from tqdm import tqdm
from datagenerator_from_list_v3 import DataGenerator3


cache_dir = './.cache'
memory = Memory(cache_dir, verbose=0, mmap_mode='r')
@memory.cache
def compute_means(filename, shape):

    data = h5py.File(filename, 'r')
    # X: time-frequency data
    X = np.array(data['X'])  # time-frequency input
    X = np.transpose(X, (2, 1, 0))  # rearrange dimension

    N = len(X)
    X = np.reshape(X, (N * shape[0], shape[1]))
    mean_X = X.mean(axis=0)
    X_sqrd = np.square(X)
    mean_X_sqrd = X_sqrd.mean(axis=0)
    return mean_X, mean_X_sqrd, N


all_bar_funcs = {
    'tqdm': lambda args: lambda x: tqdm(x, **args),
    # 'txt': lambda args: lambda x: text_progessbar(x, **args),
    'False': lambda args: iter,
    'None': lambda args: iter,
}


def ParallelExecutor(use_bar='tqdm', **joblib_args):
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type" % bar)
            return Parallel(**joblib_args)(bar_func(op_iter))
        return tmp
    return aprun


class DataGeneratorWrapper:
    def __init__(self, eeg_filelist=None, eog_filelist=None, emg_filelist=None, num_fold=1, data_shape=np.array([29, 129]), seq_len=20, shuffle=False):

        # Init params

        self.eeg_list_of_files = []
        self.eog_list_of_files = []
        self.emg_list_of_files = []
        self.file_sizes = []

        # time-frequency data shape
        self.data_shape = data_shape

        # how many folds the data is split to fold-wise loading
        self.num_fold = num_fold
        # current fold index
        self.current_fold = 0
        # subjects of the fold
        self.sub_folds = []

        self.seq_len = seq_len
        self.Ncat = 5  # five-class sleep staging

        self.shuffle = shuffle

        # list of files and their size
        if eeg_filelist is not None:
            print('Reading EEG file list')
            self.eeg_list_of_files, self.file_sizes = self.read_file_list(eeg_filelist)
        if eog_filelist is not None:
            print('Reading EOG file list')
            self.eog_list_of_files, _ = self.read_file_list(eog_filelist)
        if emg_filelist is not None:
            print('Reading EMG file list')
            self.emg_list_of_files, _ = self.read_file_list(emg_filelist)

        self.eeg_meanX, self.eeg_stdX = None, None
        self.eog_meanX, self.eog_stdX = None, None
        self.emg_meanX, self.emg_stdX = None, None

        # data generator
        self.gen = None

        #

    def read_file_list(self, filelist):
        list_of_files = []
        file_sizes = []
        with open(filelist) as f:
            lines = f.readlines()
            t = tqdm(lines)
            for l in t:
                # print(l)
                items = l.split('\t')
                t.set_description(f'{items[0][3:]}')
                list_of_files.append(items[0][3:])
                file_sizes.append(int(items[1]))
        return list_of_files, file_sizes

    def compute_eeg_normalization_params(self):
        if(len(self.eeg_list_of_files) == 0):
            return
        self.eeg_meanX, self.eeg_stdX = self.load_data_compute_norm_params(self.eeg_list_of_files)

    def compute_eog_normalization_params(self):
        if(len(self.eog_list_of_files) == 0):
            return
        self.eog_meanX, self.eog_stdX = self.load_data_compute_norm_params(self.eog_list_of_files)

    def compute_emg_normalization_params(self):
        if(len(self.eog_list_of_files) == 0):
            return
        self.emg_meanX, self.emg_stdX = self.load_data_compute_norm_params(self.emg_list_of_files)

    def set_eeg_normalization_params(self, meanX, stdX):
        self.eeg_meanX, self.eeg_stdX = meanX, stdX

    def set_eog_normalization_params(self, meanX, stdX):
        self.eog_meanX, self.eog_stdX = meanX, stdX

    def set_emg_normalization_params(self, meanX, stdX):
        self.emg_meanX, self.emg_stdX = meanX, stdX

    # read data from mat files in the list stored in the file 'filelist'
    # compute normalization parameters on the flight
    def load_data_compute_norm_params(self, list_of_files):
        meanX = None
        meanXsquared = None
        count = 0
        print('Computing normalization parameters')
        # t = tqdm(range(len(list_of_files)))
        # for i in range(len(list_of_files)):
        data = ParallelExecutor(n_jobs=1)(total=len(list_of_files))(
            delayed(compute_means)(filename=filename.strip(), shape=self.data_shape) for filename in list_of_files
        )
        meanX_i = np.array([d[0] for d in data])
        meanXsquared_i = np.array([d[1] for d in data])
        w = np.array([d[2] for d in data])
        meanX = np.average(meanX_i, axis=0, weights=w)
        meanXsquared = np.average(meanXsquared_i, axis=0, weights=w)
        varX = -np.multiply(meanX, meanX) + meanXsquared
        stdX = np.sqrt(varX * np.sum(w) / (np.sum(w) - 1))
        # print(data)
        # for i in range(len(list_of_files)):
        #     X = self.read_X_from_mat_file(list_of_files[i].strip())
        #     Ni = len(X)
        #     X = np.reshape(X,(Ni*self.data_shape[0], self.data_shape[1]))

        #     meanX_i = X.mean(axis=0)
        #     X_squared = np.square(X)
        #     meanXsquared_i = X_squared.mean(axis=0)
        #     del X

        #     if meanX is None:
        #         meanX = meanX_i
        #         meanXsquared = meanXsquared_i
        #     else:
        #         meanX = (meanX*count + meanX_i*Ni)/(count + Ni)
        #         meanXsquared = (meanXsquared*count + meanXsquared_i*Ni)/(count + Ni)
        #     count += Ni
        # varX = -np.multiply(meanX, meanX) + meanXsquared
        # stdX = np.sqrt(varX*count/(count-1))
        return meanX, stdX

    # shuffle the subjects for a new partition
    def new_subject_partition(self):
        if(self.shuffle is False):
            subject = range(len(self.file_sizes))
        else:
            subject = np.random.permutation(len(self.file_sizes))

        self.sub_folds = []
        Nsub = len(self.file_sizes)
        for i in range(self.num_fold):
            fold_i = list(range((i*Nsub)//self.num_fold, ((i+1)*Nsub)//self.num_fold))
            #fold_i = range((i*Nsub)//self.num_fold, ((i+1)*Nsub)//self.num_fold)
            # (fold_i)
            # self.sub_folds.append(subject[fold_i])
            self.sub_folds.append([subject[k] for k in fold_i])
            # print(self.sub_folds)
        self.current_fold = 0
    def read_X_from_mat_file(self, filename):
        """
        Read matfile HD5F file and parsing
        """
        # Load data
        # print(filename)
        data = h5py.File(filename, 'r')
        data.keys()
        # X: time-frequency data
        X = np.array(data['X'])  # time-frequency input
        X = np.transpose(X, (2, 1, 0))  # rearrange dimension

        return X

    def is_last_fold(self):
        return (self.current_fold == self.num_fold-1)

    def next_fold(self):
        if(self.current_fold == self.num_fold):
            self.new_subject_partition()
            self.current_fold = 0

        # at lest eeg active
        print('Current fold: ')
        print(self.current_fold)
        ind = self.sub_folds[int(self.current_fold)]
        print('N. subjects in current fold: ' + str(len(ind)))
        print('Current-fold subjects: ')
        print(ind)
        # print(ind)
        # if type(ind) is not list:
        #    ind = [ind]
        list_of_files = [self.eeg_list_of_files[int(i)] for i in ind]
        file_sizes = [self.file_sizes[int(i)] for i in ind]
        self.gen = DataGenerator3(list_of_files,
                                  file_sizes,
                                  data_shape=self.data_shape,
                                  seq_len=self.seq_len)
        self.gen.normalize(self.eeg_meanX, self.eeg_stdX)

        if(len(self.eog_list_of_files) > 0):
            list_of_files = [self.eog_list_of_files[i] for i in ind]
            eog_gen = DataGenerator3(list_of_files,
                                     file_sizes,
                                     data_shape=self.data_shape,
                                     seq_len=self.seq_len)
            eog_gen.normalize(self.eog_meanX, self.eog_stdX)

        if(len(self.emg_list_of_files) > 0):
            list_of_files = [self.emg_list_of_files[i] for i in ind]
            emg_gen = DataGenerator3(list_of_files,
                                     file_sizes,
                                     data_shape=self.data_shape,
                                     seq_len=self.seq_len)
            emg_gen.normalize(self.emg_meanX, self.emg_stdX)

        # both eog and emg not active
        if(len(self.eog_list_of_files) == 0 and len(self.emg_list_of_files) == 0):
            self.gen.X = np.expand_dims(self.gen.X, axis=-1)  # expand channel dimension
            self.gen.data_shape = self.gen.X.shape[1:]
        # 2-channel input case
        elif(len(self.eog_list_of_files) > 0 and len(self.emg_list_of_files) == 0):
            print(self.gen.X.shape)
            print(eog_gen.X.shape)
            self.gen.X = np.stack((self.gen.X, eog_gen.X), axis=-1)  # merge and make new dimension
            self.gen.data_shape = self.gen.X.shape[1:]
        # 3-channel input case
        elif(len(self.eog_list_of_files) > 0 and len(self.emg_list_of_files) > 0):
            print(self.gen.X.shape)
            print(eog_gen.X.shape)
            print(emg_gen.X.shape)
            self.gen.X = np.stack((self.gen.X, eog_gen.X, emg_gen.X), axis=-1)  # merge and make new dimension
            self.gen.data_shape = self.gen.X.shape[1:]

        if(len(self.eog_list_of_files) > 0):
            del eog_gen
        if(len(self.emg_list_of_files) > 0):
            del emg_gen

        self.current_fold += 1

        if(self.shuffle):
            self.gen.shuffle_data()
