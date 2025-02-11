import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, MaxAbsScaler, MinMaxScaler
from utils.timefeatures import time_features
import warnings
#from data_provider.m4 import M4Dataset, M4Meta

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

class CombinedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler1, scaler2):
        self.scaler1 = scaler1
        self.scaler2 = scaler2

    def fit(self, X, y=None):
        X_scaled = self.scaler1.fit_transform(X)
        self.scaler2.fit(X_scaled)
        return self

    def transform(self, X):
        X_scaled = self.scaler1.transform(X)
        return self.scaler2.transform(X_scaled)

    def inverse_transform(self, X):
        X_inv_scaled = self.scaler2.inverse_transform(X)
        return self.scaler1.inverse_transform(X_inv_scaled)

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 train_ratio=0.7, test_ratio=0.2, percent=100, 
                 force_fair_comparison_for_extendable_and_extended_input_length=False,
                 fair_comparison_length=512,
                 **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        self.force_fair_comparison_for_extendable_and_extended_input_length = force_fair_comparison_for_extendable_and_extended_input_length
        self.fair_comparison_length = fair_comparison_length

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        self.root_path = root_path
        self.data_path = data_path
        
        self.flag = flag
        self.pretrain = percent != 100
        self.percent = percent
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        cols = list(df_raw.columns)
        if self.target not in cols:
            df_raw = df_raw.rename(columns={cols[-1]:self.target})
            cols = list(df_raw.columns)
        if 'date' not in cols:
            df_raw = df_raw.rename(columns={cols[0]:'date'})
            cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        self.col_names = ['date'] + cols + [self.target]
        df_raw = df_raw[self.col_names]
        
        # This few-shot/zero-shot of ETTh1/2 is adpoted from Time-LLM
        # FROM: https://github.com/KimMeen/Time-LLM/blob/main/data_provider_pretrain/data_loader.py
        # Note: We do not exactly know why Time-LLM added 4 months of data to the training set compared to the original training data GPT4TS used. It seems that they have used this to gain a margin against GPT4TS on the ETT datasets (but not on the other datasets). Thus, we have to inherit this setting in our experiments in order to make a fair comparison.
        if self.percent != 100:
            border1s = [0, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        else:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        self.date_original = df_stamp['date'][self.seq_len:].tolist()
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_pre = data[border1:border1+self.seq_len]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.force_fair_comparison_for_extendable_and_extended_input_length and self.flag == 'test':
            bsfffc = max(0, self.seq_len - self.fair_comparison_length - index) # beginning shifting for forcing fair comparison
        else:
            bsfffc = 0
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_x = np.concatenate((np.zeros_like(self.data_x[s_begin:s_begin+bsfffc]), self.data_x[s_begin+bsfffc:s_end]), axis=0)
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = np.concatenate((np.zeros_like(self.data_stamp[s_begin:s_begin+bsfffc]), self.data_stamp[s_begin+bsfffc:s_end]), axis=0)
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 train_ratio=0.7, test_ratio=0.2, percent=100, 
                 force_fair_comparison_for_extendable_and_extended_input_length=False,
                 fair_comparison_length=512,
                 **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        self.force_fair_comparison_for_extendable_and_extended_input_length = force_fair_comparison_for_extendable_and_extended_input_length
        self.fair_comparison_length = fair_comparison_length
        
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        self.root_path = root_path
        self.data_path = data_path
        
        self.flag = flag
        self.pretrain = percent != 100
        self.percent = percent
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        cols = list(df_raw.columns)
        if self.target not in cols:
            df_raw = df_raw.rename(columns={cols[-1]:self.target})
            cols = list(df_raw.columns)
        if 'date' not in cols:
            df_raw = df_raw.rename(columns={cols[0]:'date'})
            cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        self.col_names = ['date'] + cols + [self.target]
        df_raw = df_raw[self.col_names]
        
        # This few-shot/zero-shot pretrain setting of ETTm1/2 is adpoted from Time-LLM
        # FROM: https://github.com/KimMeen/Time-LLM/blob/main/data_provider_pretrain/data_loader.py
        # Note: We do not exactly know why Time-LLM added 4 months of data to the training set compared to the original training data GPT4TS used. It seems that they have used this to gain a margin against GPT4TS on the ETT datasets (but not on the other datasets). Thus, we have to inherit this setting in our experiments in order to make a fair comparison.
        if self.pretrain:
            border1s = [0, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
                        12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
                        12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        self.date_original = df_stamp['date'][self.seq_len:].tolist()
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_pre = data[border1:border1+self.seq_len]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.force_fair_comparison_for_extendable_and_extended_input_length and self.flag == 'test':
            bsfffc = max(0, self.seq_len - self.fair_comparison_length - index) # beginning shifting for forcing fair comparison
        else:
            bsfffc = 0
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_x = np.concatenate((np.zeros_like(self.data_x[s_begin:s_begin+bsfffc]), self.data_x[s_begin+bsfffc:s_end]), axis=0)
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = np.concatenate((np.zeros_like(self.data_stamp[s_begin:s_begin+bsfffc]), self.data_stamp[s_begin+bsfffc:s_end]), axis=0)
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def append_rows_with_date(df, N):
    df['date'] = pd.to_datetime(df['date'])

    date_diff = df['date'].iloc[-1] - df['date'].iloc[-2]

    last_date = df['date'].iloc[-1]

    new_rows = []
    for i in range(1, N + 1):
        new_date = last_date + i * date_diff
        new_row = [new_date] + [0] * (df.shape[1] - 1)
        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows, columns=df.columns)
    df_extended = pd.concat([df, new_df], ignore_index=True)
    
    return df_extended

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 train_ratio=0.7, test_ratio=0.2, percent=100, 
                 force_fair_comparison_for_extendable_and_extended_input_length=False,
                 fair_comparison_length=512,
                 min_max_scaling=True,
                 do_forecasting=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.force_fair_comparison_for_extendable_and_extended_input_length = force_fair_comparison_for_extendable_and_extended_input_length
        self.fair_comparison_length = fair_comparison_length
        
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        self.root_path = root_path
        self.data_path = data_path
        
        self.flag = flag
        self.percent = percent

        # self.scale_factor = scale_factor
        self.do_forecasting = do_forecasting
        self.min_max_scaling = min_max_scaling
        self.__read_data__()

    def __read_data__(self):
        if self.min_max_scaling:
            self.scaler = MinMaxScaler() # CombinedScaler(scaler1=PowerTransformer(), scaler2=MinMaxScaler()) # MinMaxScaler() # MaxAbsScaler() # QuantileTransformer(output_distribution='normal') # PowerTransformer() #StandardScaler()
        else:
            self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        #df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw[[col for col in df_raw.columns if col == 'date' or 'date' not in col]]
        df_raw = df_raw.fillna(0)

        if self.do_forecasting:
            df_raw = append_rows_with_date(df_raw, self.pred_len)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        
        cols = list(df_raw.columns)
        if self.target not in cols:
            df_raw = df_raw.rename(columns={cols[-1]:self.target})
            cols = list(df_raw.columns)
        if 'date' not in cols:
            df_raw = df_raw.rename(columns={cols[0]:'date'})
            cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        self.col_names = ['date'] + cols + [self.target]
        df_raw = df_raw[self.col_names]
        # print(cols)
        num_train = int(len(df_raw) * self.train_ratio)
        num_test = int(len(df_raw) * self.test_ratio)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            # self.scaler.scale_ = self.scaler.scale_ * self.scale_factor
            # self.scaler.var_ = np.square(self.scaler.scale_)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        self.date_original = df_stamp['date'][self.seq_len:].tolist()
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_pre = data[border1:border1+self.seq_len]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.force_fair_comparison_for_extendable_and_extended_input_length and self.flag == 'test':
            bsfffc = max(0, self.seq_len - self.fair_comparison_length - index) # beginning shifting for forcing fair comparison
        else:
            bsfffc = 0
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_x = np.concatenate((np.zeros_like(self.data_x[s_begin:s_begin+bsfffc]), self.data_x[s_begin+bsfffc:s_end]), axis=0)
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = np.concatenate((np.zeros_like(self.data_stamp[s_begin:s_begin+bsfffc]), self.data_stamp[s_begin+bsfffc:s_end]), axis=0)
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', *args, **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        self.col_names = df.columns[1:].tolist()
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        
        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', *args, **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)
        self.col_names = df_raw.columns[1:].tolist()

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None,
                 **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        
        self.date_original = df_stamp['date'][self.seq_len:].tolist()
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
