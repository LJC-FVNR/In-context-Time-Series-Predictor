import random
import warnings
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from tqdm import tqdm
from collections import Counter
import copy
from scipy.signal import resample
from data_provider.ts_feature import extract_time_series_features, time_series_transformation

class InContextTimeSeriesTokenizer:
    def __init__(self, configs):
        super().__init__()
        #self.configs = configs
        self.task_id_mapping = {"forecasting": 0, "classification": 1, "imitation": 2, "imputation": 3, 
                                "cropping": 4, "reflection": 5, "shifting": 6, "hyperres": 7, 
                                "statistics": 8, "differencing": 9, "movingavg":10, "expsmoothing": 11, "decomposition": 12}
        self.stage = getattr(configs, "stage", "dev") # "dev" for training and evaluation; "inference" for real-world prediction
        
        self.max_L_I = getattr(configs, "max_L_I", 65536)      # max time series input length
        self.lookback = getattr(configs, "lookback", 1536)     # this will be the input dimension of the X part for the embedding layer
        self.future = getattr(configs, "future", 1536) # forecasting horizon of the forecasting task

        self.shorter_lookback_for_finetuning = getattr(configs, "shorter_lookback_for_finetuning", None)
        self.shorter_future_for_finetuning = getattr(configs, "shorter_future_for_finetuning", None)
        
        self.randomized_training_flag = getattr(configs, "randomized_training_flag", False)  # be careful! this will randomly shorten the lookback and future length in the data processing
        
        self.random_series_shuffle = getattr(configs, "random_series_shuffle", True)         # randomly shuffle the order of the series during the training
        
        self.zero_padding_to_hard_token_limit = getattr(configs, "zero_padding_to_hard_token_limit", True)
        
        self.force_resize_time_series_to_size_limit = getattr(configs, "force_resize_time_series_to_size_limit", False)   # 
        
        self.soft_token_limit = getattr(configs, "soft_token_limit", 8192)  # in randomized training, how much tokens generated from each series is allowed
        self.hard_token_limit = getattr(configs, "hard_token_limit", 8192)
        
        self.sampling_step = getattr(configs, "sampling_step", 8)    # sampling stride m in forecasting task
        
        # The Padding Index of Channel Label is 0
        self.max_channel_vocab_size = getattr(configs, "max_channel_vocab_size", 4096)
        
        # The Padding Index of Position Label is 0
        self.max_position_vocab_size = getattr(configs, "max_position_vocab_size", 4096)
        
        self.max_source_vocab_size = getattr(configs, "max_source_vocab_size", 8192)
        
        # tag: multi-label multi-hot space
        self.max_tag_vocab_size = getattr(configs, "max_tag_vocab_size", 4096)
        # self.tag_vocab_encoder = MultiLabelBinarizer()
        # self.tag_vocab_encoder.fit(np.arange(self.max_tag_vocab_size).reshape(-1, 1))
        
        self.number_of_targets = getattr(configs, "number_of_targets", 0)      # if not 0, only the last [number_of_targets] series will be involved in the predicting tokens
        
        self.n_series_registrar = getattr(configs, "n_series_registrar", 0)
        
        self.cls_max_random_number_of_targets = getattr(configs, "cls_max_random_number_of_targets", 8)
    
    def tokenize(self, x, y=None, target_series=None, task="forecasting", force_mask=None, force_p=None, **kwargs):
        if task == "forecasting":
            return self._tokenize_forecasting(x, **kwargs)
        elif task == "classification":
            assert y is not None, "classification requires y labels as input"
            return self._tokenize_classification(x, y, target_series=target_series, **kwargs)
        elif task == "imitation":
            assert y is not None, "imitation requires y input"
            return self._tokenize_imitation(x, y, target_series=target_series, **kwargs)
        elif task == "imputation":
            return self._tokenize_imputation(x, y=y, target_series=target_series, force_mask=force_mask, force_p=force_p, **kwargs)
        elif task == "cropping":
            return self._tokenize_cropping(x, **kwargs)
        elif task == "reflection":
            return self._tokenize_reflection(x, **kwargs)
        elif task == "shifting":
            return self._tokenize_shifting(x, **kwargs)
        elif task == "hyperres":
            return self._tokenize_hyperres(x, y=y, target_series=target_series, **kwargs)
        elif task == "statistics":
            return self._tokenize_statistics(x, y=y, target_series=target_series, **kwargs)
        elif task in ["differencing", "movingavg", "expsmoothing", "decomposition"]:
            return self._tokenize_ts_transformation(x, y=y, target_series=target_series, method=task, **kwargs)
        
        raise NotImplementedError(f"Task: {task}")
    
    def _tokenize_forecasting(self, x, force_lookback=None, force_future=None):
        """
        input: training - (C, L) or inference - (C, L_I)
        output: [x{N_tokens, max_lookback}, y_true{C, L_P}, y_input{N_tokens, max_L_P}, 
             source_label{N_tokens, 1(LongTensor)}, task_label{1}]
        """
        if x.shape[0] > self.max_position_vocab_size:
            x = x[0:self.max_position_vocab_size - 1]
        # -1. (C, L) to (L, C)
        x = x.T
        # 0. if training, use the original last (L_P, C) as y_true{L_P, C} and (L_I, C) as x_i, else, all as x_i
        C = x.shape[1]
        current_L = x.shape[0]         # L = L_I + L_P, lookback < L_I-L_P, L_I > lookback+L_P, L_I > lookback+future, L_I = lookback+future+Ns, L = lookback+2*future+Ns
    
        if self.stage == "dev":
            if self.random_series_shuffle:
                x = self.shuffle_array_along_axis(x, axis=1)

            if self.randomized_training_flag:
                input_lookback = self.lookback if self.shorter_lookback_for_finetuning is None else self.shorter_lookback_for_finetuning
                input_future = self.future if self.shorter_future_for_finetuning is None else self.shorter_future_for_finetuning
                lookback, future, n_samples = self.get_valid_randomized_length(current_L, input_lookback, input_future, 
                                                                               self.sampling_step*self.soft_token_limit/C)
            x_input, y_true = x[0:-future, :], x[-future:, :]

        elif self.stage == "inference":
            lookback = self.lookback if force_lookback is None else force_lookback
            future = self.future if force_future is None else force_future
            x_input = x
            y_true = np.full((future, x.shape[1]), np.nan, dtype=np.float32)
            
        else:
            raise NotImplementedError(f'Stage [{self.stage}] is not implemented.')
            
        y_true = y_true if self.number_of_targets == 0 else y_true[:, -self.number_of_targets:]
        
        # The y_true is padded to max future length!!! mask out the padded part in the output of the model results!
        # if self.zero_padding:
        #     padding_y = ((0, self.future - y_true.shape[0]), (0, 0))
        #    y_true = np.pad(y_true, padding_y, mode='constant', constant_values=np.nan)
        
        L_I = x_input.shape[0]
        valid_input = L_I >= lookback + future
        assert valid_input, f"L >= lookback + 2*future is not fulfilled for the current input (L={current_L}, lookback={lookback}, future={future})"

        safe_token_group_limit = self.hard_token_limit // C + 1
        
        # 1. use (L_I, C) and lookback to generate {N_tokens-C, lookback+L_P}
        token_context = self.time_series_to_tokens(x_input.T, length=lookback+future, stride=self.sampling_step)  # N-1 C d
        token_context = token_context[-safe_token_group_limit:]
        token_context = token_context.reshape(-1, token_context.shape[-1])                # (N-1)*C d
        
        # 2. split last (L_I, C)[-lookback:, :].T, pad with L_P length of float("-inf") to get {C, lookback+L_P}
        future_padding = np.full((C, future), float("inf"), dtype=np.float32)
        token_target = np.concatenate([x_input[-lookback:, :].T, future_padding], axis=-1)                        # C d
        token_target = token_target if self.number_of_targets == 0 else token_target[-self.number_of_targets:, :]
        
        # 3. concatenate to get {N_tokens, lookback+L_P}
        tokens = np.concatenate([token_context, token_target], axis=0)                                            # n d=lb+ft
        
        # 3.5 Split lookback and future
        token_x_part = tokens[:, 0:lookback]
        token_y_part = tokens[:, -future:]
        
        # 4. pad with NaN to get {N_tokens, max_lookback+max_L_P=input_dim}
        x_pad_len = self.lookback - lookback
        x_pad = np.full((token_x_part.shape[0], x_pad_len), np.nan, dtype=np.float32)
        token_x_part = np.concatenate([x_pad, token_x_part], axis=-1)                   # n_input max_lookback
        
        y_pad_len = self.future - future
        y_pad = np.full((token_y_part.shape[0], y_pad_len), np.nan, dtype=np.float32)
        token_y_part = np.concatenate([token_y_part, y_pad], axis=-1)                   # n_target max_future
        
        # 5. generate channel source label with C: {N_tokens-C, 1} + {C, 1}
        # channel label
        channel_label = np.linspace(1+np.random.randint(0, 10), self.max_channel_vocab_size, C, dtype=int)
        channel_label = np.flip(channel_label, axis=0).copy()
        channel_label = np.tile(channel_label, token_context.shape[0]//C + 1)
        channel_label_context = channel_label[0:token_context.shape[0]]
        channel_label_target = channel_label[-C:] if self.number_of_targets == 0 else channel_label[-self.number_of_targets:]
        channel_label = np.concatenate([channel_label_context, channel_label_target])
        
        position_label = np.linspace(1+np.random.randint(0, 10), self.max_position_vocab_size, token_context.shape[0]//C + 1, dtype=int)
        position_label = np.flip(position_label, axis=0).copy()
        position_label = np.repeat(position_label, C)
        position_label_context = position_label[0:token_context.shape[0]]
        position_label_target = position_label[-C:] if self.number_of_targets == 0 else position_label[-self.number_of_targets:]
        position_label = np.concatenate([position_label_context, position_label_target])
        
        # source label is disabled
        source_label = np.zeros(position_label.shape[0], dtype=int)                        # full 0
        
        # additional tag space is disabled
        tag_multihot = np.zeros((position_label.shape[0], self.max_tag_vocab_size), dtype=int)            # n max_tag_vocab_size
        
        # pad to same token length
        if token_x_part.shape[0] < self.hard_token_limit and self.zero_padding_to_hard_token_limit:
            padding_len = ((self.hard_token_limit - token_x_part.shape[0], 0), (0, 0))
            padding_len_1d = (self.hard_token_limit - token_x_part.shape[0], 0)
            token_x_part = np.pad(token_x_part, padding_len, mode='constant', constant_values=0)
            token_y_part = np.pad(token_y_part, padding_len, mode='constant', constant_values=0)
            channel_label = np.pad(channel_label, padding_len_1d, mode='constant', constant_values=0)
            position_label = np.pad(position_label, padding_len_1d, mode='constant', constant_values=0)
            source_label = np.pad(source_label, padding_len_1d, mode='constant', constant_values=0)
            tag_multihot = np.pad(tag_multihot, padding_len, mode='constant', constant_values=0)
        
        token_x_part = token_x_part[-self.hard_token_limit:]
        token_y_part = token_y_part[-self.hard_token_limit:]
        channel_label = channel_label[-self.hard_token_limit:]
        position_label = position_label[-self.hard_token_limit:]
        source_label = source_label[-self.hard_token_limit:]
        tag_multihot = tag_multihot[-self.hard_token_limit:]
        
        y_true = y_true.T
        
        # for the output, all of the channel dimension should be flipped since the nested tensor padding method only allows padding on the bottom right, which means that we need to flip here and flip back after the nested padding to get the desired "float to the right" token format - the temporal dimension need not to be flipped since its inherently "float to the left"
        return (token_x_part[::-1].copy(),    # N max_lookback
                y_true[::-1].copy(),         # C future
                token_y_part[::-1].copy(),   # N max_future
                channel_label[::-1].copy(),  # N
                position_label[::-1].copy(), # N
                source_label[::-1].copy(),   # N
                tag_multihot[::-1].copy(),   # N max_tag_vocab_size
                np.array(list(y_true.shape)),   # output shape
                self.task_id_mapping["forecasting"]  # task label
                )    
                
    def _tokenize_classification(self, x, y, target_series=None):
        """
        training -
            TS1    y1           from datapoint1
            TS2    y1           from datapoint1
            TS3    y1           from datapoint1
            TS4    y1           from datapoint1
            
            TS5    y2           from datapoint2
            TS6    y2           from datapoint2
            TS7    y2           from datapoint2
            
            TS8    y2           from datapoint3
            TS9    y2           from datapoint3
            
            TS10   y3           from datapoint4
            TS11   y3           from datapoint4
            
            TS12   y1           from datapoint5
            TS13   y1           from datapoint5
            
            TS14   y2           from datapoint6           
            
            TS15   y3           from datapoint7
        x: list of C L [np.array(C, L), np.array(C, L), ..., ], N C L
        y: list (N, )
        target_series: (C, L), used in inference
        """
        assert len(x) == len(y), "The context input should be correctly paired"
        
        if self.force_resize_time_series_to_size_limit:
            for i in range(len(x)):
                x[i] = self.resize_time_series(x[i], target_length=self.lookback)
                target_series = self.resize_time_series(target_series, target_length=self.lookback) if target_series is not None else None
        
        y_count = Counter(y)
        if self.stage == "dev":
            if self.random_series_shuffle:
                indices = list(range(len(x)))
                random.shuffle(indices)
                x = [x[index] for index in indices]
                y = [y[index] for index in indices]
                
            y_for_training = [category for category, count in y_count.items() if count >= 2]
            assert y_for_training, "Randomized Classification Tokenization: There should be at least one category that have more than 1 datapoints."
            selected_y_label = np.random.choice(y_for_training)
            
            selected_y_label_index_list = [index for index, y_label in enumerate(y) if y_label == selected_y_label]
            selected_y_label_index = np.random.choice(selected_y_label_index_list)
            selected_y_item = y[selected_y_label_index]
            
            selected_x = x.pop(selected_y_label_index)
            x.append(selected_x)
            selected_y = y.pop(selected_y_label_index)
            y.append("[Target]")
            
            pred_C, pred_L = selected_x.shape

        elif self.stage == 'inference':
            assert target_series is not None, "Classification inference mode needs at least one input"
            pred_C, pred_L = target_series.shape
            x.append(target_series)
            y.append("[Target]")
            
        else:
            raise NotImplementedError(f'Stage [{self.stage}] is not implemented.')

        # Transform list of 2d time series datapoints to 2D array
        max_len = max([i.shape[1] for i in x])
        total_channels = sum([i.shape[0] for i in x])
        x_input, y_input, dp_label = np.zeros((total_channels, max_len), dtype=np.float32), [], np.zeros(total_channels, dtype=int)
        
        dp_label_generated = np.linspace(1+np.random.randint(0, 32), self.max_source_vocab_size, len(x), dtype=int)
        dp_label_generated = np.flip(dp_label_generated, axis=0).copy()

        index = 0
        for i, current_dp in enumerate(x):
            for current_series in current_dp:
                x_input[index, -len(current_series):] = current_series
                y_input.append(y[i])
                dp_label[index] = dp_label_generated[i]
                index += 1

        x_input = x_input[:, -self.max_L_I:]
        
        x, y = x_input, y_input
        
        le_y = LabelEncoder()
        y = np.array(le_y.fit_transform(["[Pad]", "[Target]"] + y))[2:]   # C_all,
        
        # construct target
        if self.stage == "dev":
            y_target_id = le_y.transform([selected_y_item])
            y_true = np.array([y_target_id[0] for _ in range(pred_C)])                   # pred_C,

        if x.shape[1] < self.lookback:
            x_pad_len = self.lookback - x.shape[1]
            x_pad = np.full((x.shape[0], x_pad_len), np.nan, dtype=np.float32)
            token_x_part = np.concatenate([x_pad, x], axis=-1)                   # C_all lookback
            N = 1
        else:
            token_x_part = self.time_series_to_tokens(x, length=self.lookback, stride=self.lookback) # N C_all lookback, Non-overlapping split
            N = token_x_part.shape[0]
            y = np.repeat(y, N)
            dp_label = np.repeat(dp_label, N)
            token_x_part = token_x_part.reshape(-1, self.lookback)               # N*C_all lookback
        
        # generate channel&position label
        C = total_channels
        channel_label = np.linspace(1+np.random.randint(0, 10), self.max_channel_vocab_size, C, dtype=int)
        channel_label = np.flip(channel_label, axis=0).copy()
        channel_label = np.tile(channel_label, N)
        
        position_label = np.linspace(1, self.max_position_vocab_size, N, dtype=int)
        position_label = np.flip(position_label, axis=0).copy()
        position_label = np.repeat(position_label, C)
        
        source_label = dp_label
        
        # additional tag space is disabled
        tag_multihot = np.zeros((position_label.shape[0], self.max_tag_vocab_size), dtype=int)            # n max_tag_vocab_size
        
        token_y_part = y
        
        # pad to same token length
        if token_x_part.shape[0] < self.hard_token_limit and self.zero_padding_to_hard_token_limit:
            # padding_len = ((self.hard_token_limit - token_x_part.shape[0], 0), (0, 0))
            # padding_len_1d = (self.hard_token_limit - token_x_part.shape[0], 0)
            # token_x_part = np.pad(token_x_part, padding_len, mode='constant', constant_values=0)
            # token_y_part = np.pad(token_y_part, padding_len_1d, mode='constant', constant_values=0)   # 1 dimension
            # channel_label = np.pad(channel_label, padding_len_1d, mode='constant', constant_values=0)
            # position_label = np.pad(position_label, padding_len_1d, mode='constant', constant_values=0)
            # source_label = np.pad(source_label, padding_len_1d, mode='constant', constant_values=0)
            # tag_multihot = np.pad(tag_multihot, padding_len, mode='constant', constant_values=0)
            token_x_part_padded = np.full((self.hard_token_limit, self.lookback), np.nan, dtype=np.float32)
            #token_y_part_padded = np.full((self.hard_token_limit, self.future), np.nan, dtype=np.float32)
            channel_label_padded = np.zeros(self.hard_token_limit, dtype=int)
            position_label_padded = np.zeros(self.hard_token_limit, dtype=int)
            source_label_padded = np.zeros(self.hard_token_limit, dtype=int)
            tag_multihot_padded = np.zeros((self.hard_token_limit, self.max_tag_vocab_size), dtype=int)

            token_x_part_padded[:token_x_part.shape[0], x_pad_len:] = token_x_part
            #token_y_part_padded[:token_y_part.shape[0], :future] = token_y_part
            channel_label_padded[:channel_label.shape[0]] = channel_label
            position_label_padded[:position_label.shape[0]] = position_label
            source_label_padded[:source_label.shape[0]] = source_label
            tag_multihot_padded[:tag_multihot.shape[0], :] = tag_multihot

            token_x_part = token_x_part_padded
            padding_len_1d = (self.hard_token_limit - token_x_part.shape[0], 0)
            token_y_part = np.pad(token_y_part, padding_len_1d, mode='constant', constant_values=0)   # 1 dimension
            channel_label = channel_label_padded
            position_label = position_label_padded
            source_label = source_label_padded
            tag_multihot = tag_multihot_padded
        
        token_x_part = token_x_part[-self.hard_token_limit:]
        token_y_part = token_y_part[-self.hard_token_limit:]
        channel_label = channel_label[-self.hard_token_limit:]
        position_label = position_label[-self.hard_token_limit:]
        source_label = source_label[-self.hard_token_limit:]
        tag_multihot = tag_multihot[-self.hard_token_limit:]
        
        return (token_x_part[::-1].copy(),    # N max_lookback
                y_true[::-1].copy(),         # N
                token_y_part[::-1].copy(),   # N
                channel_label[::-1].copy(),  # N
                position_label[::-1].copy(), # N
                source_label[::-1].copy(),   # N
                tag_multihot[::-1].copy(),   # N max_tag_vocab_size
                np.array(list(y_true.shape)),   # output shape
                self.task_id_mapping["classification"]  # task label
                )    
    
    def _tokenize_imitation(self, x, y, target_series=None, do_resize=True):
        """
        training -
        x: list of (C, L) [np.array(C, L), np.array(C, L), ..., ], N C L
        y: list of (C, L) [np.array(C, L), np.array(C, L), ..., ], N C L
        inference -
        target_series: np.array(C, L)
        """
        
        x_total_channels = sum([i.shape[0] for i in x])
        y_total_channels = sum([i.shape[0] for i in y])
        
        assert len(x) == len(y) and x_total_channels == y_total_channels, "The context input should be correctly paired"

        series_limit = min(self.max_position_vocab_size, self.max_source_vocab_size)
        if x_total_channels > series_limit:
            diff = x_total_channels - series_limit
            x[-1] = x[-1][0:-diff]
            y[-1] = y[-1][0:-diff]
            x_total_channels = sum([i.shape[0] for i in x])
            y_total_channels = sum([i.shape[0] for i in y])
        
        # skip if upstream processes have already done the resizing with `do_resize`
        if self.force_resize_time_series_to_size_limit and do_resize:
            for i in range(len(x)):
                x[i] = self.resize_time_series(x[i], target_length=self.lookback)
                y[i] = self.resize_time_series(y[i], target_length=self.future)
            target_series = self.resize_time_series(target_series, target_length=self.lookback) if target_series is not None else None
        
        elif not self.force_resize_time_series_to_size_limit and do_resize:
            # check the length of the input
            len_limit = min(self.lookback, self.future)
            for i in range(len(x)):
                x[i] = x[i][:, -len_limit:]
                y[i] = y[i][:, -len_limit:]
            target_series = target_series[:, -len_limit:] if target_series is not None else None
        
        if self.stage == "dev":
            # the _tokenize_imitation method is used as a downstream processing method for other tasks: multiple random shuffling should be prevented
            if self.random_series_shuffle:
                indices = list(range(len(x)))
                random.shuffle(indices)
                x = [x[index] for index in indices]
                y = [y[index] for index in indices]
            
            selected_index = np.random.choice(range(len(x)))
            selected_x = x.pop(selected_index)
            x.append(selected_x)
            selected_y = y.pop(selected_index)
            y_true = selected_y
            y.append(np.full((selected_y.shape[0], selected_y.shape[1]), float("inf"), dtype=np.float32))
        elif self.stage == "inference":
            assert target_series is not None, "Inference mode requires at least one target input series"
        else:
            raise NotImplementedError(f'Stage [{self.stage}] is not implemented.')

        max_len_x = max([i.shape[1] for i in x])
        max_len_x = min(max_len_x, self.lookback)
        max_len_y = max([i.shape[1] for i in y])
        max_len_y = min(max_len_y, self.future)

        x_input, y_input = np.full((x_total_channels, self.lookback), np.nan, dtype=np.float32), np.full((y_total_channels, self.future), np.nan, dtype=np.float32)
        
        index = 0
        for current_x, current_y in zip(x, y):
            for current_x_series, current_y_series in zip(current_x, current_y):
                cut_len = min(len(current_x_series), self.lookback)
                x_input[index, -cut_len:] = current_x_series[-cut_len:]
                cut_len = min(len(current_y_series), self.future)
                y_input[index, 0:cut_len] = current_y_series[0:cut_len]
                index += 1
                
        token_x_part = x_input # C_all Lookback
        token_y_part = y_input # C_all future
        
        # channel label is disabled
        channel_label = np.zeros(token_x_part.shape[0], dtype=int)                        # full 0
        
        # potision label is disabled
        position_label = np.linspace(1, self.max_position_vocab_size, token_x_part.shape[0], dtype=int)#np.arange(token_x_part.shape[0]) + 1                        # full 0
        
        # source label is used to mark the source of each series from the input datapoints/datasets
        dp_label = np.zeros(x_total_channels, dtype=int)
        dp_label_generated = np.linspace(1+np.random.randint(0, 32), self.max_source_vocab_size, len(x), dtype=int)
        dp_label_generated = np.flip(dp_label_generated, axis=0).copy()
        index = 0
        for i, current_dp in enumerate(x):
            for current_series in current_dp:
                dp_label[index] = dp_label_generated[i]
                index += 1
        source_label = dp_label
        
        
        # additional tag space is disabled
        tag_multihot = np.zeros((token_x_part.shape[0], self.max_tag_vocab_size), dtype=int)            # n max_tag_vocab_size
        
        # pad to same token length
        if token_x_part.shape[0] < self.hard_token_limit and self.zero_padding_to_hard_token_limit:
            padding_len = ((self.hard_token_limit - token_x_part.shape[0], 0), (0, 0))
            padding_len_1d = (self.hard_token_limit - token_x_part.shape[0], 0)
            token_x_part = np.pad(token_x_part, padding_len, mode='constant', constant_values=0)
            token_y_part = np.pad(token_y_part, padding_len, mode='constant', constant_values=0)
            channel_label = np.pad(channel_label, padding_len_1d, mode='constant', constant_values=0)
            position_label = np.pad(position_label, padding_len_1d, mode='constant', constant_values=0)
            source_label = np.pad(source_label, padding_len_1d, mode='constant', constant_values=0)
            tag_multihot = np.pad(tag_multihot, padding_len, mode='constant', constant_values=0)

        else:
            token_x_part = token_x_part[-self.hard_token_limit:]
            token_y_part = token_y_part[-self.hard_token_limit:]
            channel_label = channel_label[-self.hard_token_limit:]
            position_label = position_label[-self.hard_token_limit:]
            source_label = source_label[-self.hard_token_limit:]
            tag_multihot = tag_multihot[-self.hard_token_limit:]
        
        return (token_x_part[::-1].copy(),    # N max_lookback
                y_true[::-1].copy(),         # 
                token_y_part[::-1].copy(),   # N
                channel_label[::-1].copy(),  # N
                position_label[::-1].copy(), # N
                source_label[::-1].copy(),   # N
                tag_multihot[::-1].copy(),   # N max_tag_vocab_size
                np.array(list(y_true.shape)),   # output shape
                self.task_id_mapping["imitation"]  # task label
                )    

    def _tokenize_imputation(self, x, y=None, target_series=None, force_mask=None, force_p=None):
        """
        training - 
        x: list of (C, L) [np.array(C, L), np.array(C, L), ..., ], N C L
        force_mask (optional): list of (L,) with mask like [0, 0, np.nan, np.nan, 0, 0]
        """
        if self.force_resize_time_series_to_size_limit:
            for i in range(len(x)):
                x[i] = self.resize_time_series(x[i], target_length=self.lookback) 
                if y is not None:
                    y[i] = self.resize_time_series(y[i], target_length=self.lookback)
            target_series = self.resize_time_series(target_series, target_length=self.lookback) if target_series is not None else None
        else:
            # check the length of the input
            len_limit = min(self.lookback, self.future)
            for i in range(len(x)):
                x[i] = x[i][:, -len_limit:]
                if y is not None:
                    y[i] = y[i][:, -len_limit:]
            target_series = target_series[:, -len_limit:] if target_series is not None else None
        
        y = copy.deepcopy(x) if y is None else y
        if self.stage == "dev":
            if force_mask is None:
                for index in range(len(x)):
                    p = np.random.uniform(0.05, 0.90) if force_p is None else force_p
                    if np.random.rand() > 0.5:
                        mask = self.generate_random_mask(x[index].shape[1], p)
                    else:
                        mask = self.generate_continuous_mask(x[index].shape[1], p)
                    x[index] = x[index] + mask
            else:
                for index in range(len(x)):
                    x[index] = x[index] + force_mask[index]
            
        elif self.stage == "inference":
            assert target_series is not None, "Inference mode requires at least one target input series"
            if force_mask is not None:
                for index in range(len(x)):
                    x[index] = x[index] + force_mask[index]
        else:
            raise NotImplementedError(f'Stage [{self.stage}] is not implemented.')
            
        token_x_part, y_true, token_y_part, channel_label, position_label, source_label, tag_multihot, _, _ = self._tokenize_imitation(x, y, target_series=target_series, do_resize=False)
        
        return (token_x_part[::-1].copy(),    # N max_lookback
                y_true[::-1].copy(),         # 
                token_y_part[::-1].copy(),   # N
                channel_label[::-1].copy(),  # N
                position_label[::-1].copy(), # N
                source_label[::-1].copy(),   # N
                tag_multihot[::-1].copy(),   # N max_tag_vocab_size
                np.array(list(y_true.shape)),   # output shape
                self.task_id_mapping["imputation"]  # task label
                )    
        
    def _tokenize_cropping(self, x, target_series=None):
        """
        training -
        x: list of (C, L) [np.array(C, L), np.array(C, L), ..., ], N C L
        not for inference
        """
        
        if self.force_resize_time_series_to_size_limit:
            for i in range(len(x)):
                x[i] = self.resize_time_series(x[i], target_length=self.lookback) 
            target_series = self.resize_time_series(target_series, target_length=self.lookback) if target_series is not None else None
        else:
            # check the length of the input
            len_limit = min(self.lookback, self.future)
            for i in range(len(x)):
                x[i] = x[i][:, -len_limit:]
            target_series = target_series[:, -len_limit:] if target_series is not None else None
                
        # check the length of the input
        len_limit = min(self.lookback, self.future)
        for i in range(len(x)):
            x[i] = x[i][:, -len_limit:]
        
        y = copy.deepcopy(x)
        
        min_len = min([i.shape[1] for i in x])

        start_point = np.random.randint(0, min_len-2)
        end_point = np.random.randint(start_point+1, min_len)
        
        for index in range(len(y)):
            y[index] = y[index][:, start_point:end_point]
        
        token_x_part, y_true, token_y_part, channel_label, position_label, source_label, tag_multihot, _, _ = self._tokenize_imitation(x, y, target_series=target_series, do_resize=False)
        
        return (token_x_part[::-1].copy(),    # N max_lookback
                y_true[::-1].copy(),         #
                token_y_part[::-1].copy(),   # N
                channel_label[::-1].copy(),  # N
                position_label[::-1].copy(), # N
                source_label[::-1].copy(),   # N
                tag_multihot[::-1].copy(),   # N max_tag_vocab_size
                np.array(list(y_true.shape)),   # output shape
                self.task_id_mapping["cropping"]  # task label
                )    
    
    def _tokenize_reflection(self, x, target_series=None):
        """
        training -
        x: list of (C, L) [np.array(C, L), np.array(C, L), ..., ], N C L
        not for inference
        """
        if self.force_resize_time_series_to_size_limit:
            for i in range(len(x)):
                x[i] = self.resize_time_series(x[i], target_length=self.lookback) 
            target_series = self.resize_time_series(target_series, target_length=self.lookback) if target_series is not None else None
        else:
            # check the length of the input
            len_limit = min(self.lookback, self.future)
            for i in range(len(x)):
                x[i] = x[i][:, -len_limit:]
            target_series = target_series[:, -len_limit:] if target_series is not None else None
                
        y = copy.deepcopy(x)
        
        for index in range(len(y)):
            y[index] = np.flip(y[index], axis=1).copy()
            
        token_x_part, y_true, token_y_part, channel_label, position_label, source_label, tag_multihot, _, _ = self._tokenize_imitation(x, y, target_series=target_series, do_resize=False)
        
        return (token_x_part,    # N max_lookback
                y_true,         # future, C
                token_y_part,   # N
                channel_label,  # N
                position_label, # N
                source_label,   # N
                tag_multihot,   # N max_tag_vocab_size
                np.array(list(y_true.shape)),   # output shape
                self.task_id_mapping["reflection"]  # task label
                )    
    
    def _tokenize_shifting(self, x, target_series=None):
        """
        training -
        x: list of (C, L) [np.array(C, L), np.array(C, L), ..., ], N C L
        not for inference
        """
        if self.force_resize_time_series_to_size_limit:
            for i in range(len(x)):
                x[i] = self.resize_time_series(x[i], target_length=self.lookback) 
            target_series = self.resize_time_series(target_series, target_length=self.lookback) if target_series is not None else None
        else:
            # check the length of the input
            len_limit = min(self.lookback, self.future)
            for i in range(len(x)):
                x[i] = x[i][:, -len_limit:]
            target_series = target_series[:, -len_limit:] if target_series is not None else None
                
        y = copy.deepcopy(x)
        
        min_len = min([i.shape[1] for i in x])
        shift_step = np.random.randint(1, min_len)
        circular = np.random.rand() > 0.5
        sign = 1 if np.random.rand() > 0.5 else -1
        for index in range(len(y)):
            
            y[index] = self.shift_array(y[index], sign*shift_step, 1, circular)
            
        token_x_part, y_true, token_y_part, channel_label, position_label, source_label, tag_multihot, _, _ = self._tokenize_imitation(x, y, target_series=target_series, do_resize=False)
        
        return (token_x_part[::-1].copy(),    # N max_lookback
                y_true[::-1].copy(),         # 
                token_y_part[::-1].copy(),   # N
                channel_label[::-1].copy(),  # N
                position_label[::-1].copy(), # N
                source_label[::-1].copy(),   # N
                tag_multihot[::-1].copy(),   # N max_tag_vocab_size
                np.array(list(y_true.shape)),   # output shape
                self.task_id_mapping["shifting"]  # task label
                )    
    
    def _tokenize_hyperres(self, x, y=None, target_series=None):
        """
        training -
        x: list of (C, L) [np.array(C, L), np.array(C, L), ..., ], N C L
        prepare your ground truth hyper-resolution data before using in inference mode
        """
        if self.force_resize_time_series_to_size_limit:
            for i in range(len(x)):
                x[i] = self.resize_time_series(x[i], target_length=self.lookback)
                if y is not None:
                    y[i] = self.resize_time_series(y[i], target_length=self.lookback)
            target_series = self.resize_time_series(target_series, target_length=self.lookback) if target_series is not None else None
        else:
            # check the length of the input
            len_limit = min(self.lookback, self.future)
            for i in range(len(x)):
                x[i] = x[i][:, -len_limit:]
                if y is not None:
                    y[i] = y[i][:, -len_limit:]
            target_series = target_series[:, -len_limit:] if target_series is not None else None
        
        # limit check
        for index in range(len(x)):
            if x[index].shape[1] > self.future:
                x[index] = self.resize_time_series(x[index], self.future)
        
        y = copy.deepcopy(x) if y is None else y
        
        if self.stage == "dev":
            downsampling_ratio = np.random.uniform(1, 32)
            for index in range(len(x)):
                x[index] = self.resize_time_series(x[index], int(np.ceil(x[index].shape[1]/downsampling_ratio)))
                
        token_x_part, y_true, token_y_part, channel_label, position_label, source_label, tag_multihot, _, _ = self._tokenize_imitation(x, y, target_series=target_series, do_resize=False)
        
        return (token_x_part[::-1].copy(),    # N max_lookback
                y_true[::-1].copy(),         # 
                token_y_part[::-1].copy(),   # N
                channel_label[::-1].copy(),  # N
                position_label[::-1].copy(), # N
                source_label[::-1].copy(),   # N
                tag_multihot[::-1].copy(),   # N max_tag_vocab_size
                np.array(list(y_true.shape)),   # output shape
                self.task_id_mapping["hyperres"]  # task label
                )    
        
    def _tokenize_statistics(self, x, y=None, target_series=None):
        """
        training -
        x: list of (C, L) [np.array(C, L), np.array(C, L), ..., ], N C L
        prepare your ground truth of target statistics before using in inference mode
        """
        if self.force_resize_time_series_to_size_limit:
            for i in range(len(x)):
                x[i] = self.resize_time_series(x[i], target_length=self.lookback)
                if y is not None:
                    y[i] = self.resize_time_series(y[i], target_length=self.lookback)
            target_series = self.resize_time_series(target_series, target_length=self.lookback) if target_series is not None else None
        else:
            # check the length of the input
            len_limit = min(self.lookback, self.future)
            for i in range(len(x)):
                x[i] = x[i][:, -len_limit:]
                if y is not None:
                    y[i] = y[i][:, -len_limit:]
            target_series = target_series[:, -len_limit:] if target_series is not None else None

        if y is None:
            indices = extract_time_series_features(x[0], randomized=True, get_indices=True)
            y = [np.clip(extract_time_series_features(xi, indices=indices), -20, 20) for xi in x]
        
        token_x_part, y_true, token_y_part, channel_label, position_label, source_label, tag_multihot, _, _ = self._tokenize_imitation(x, y, target_series=target_series, do_resize=False)
        
        return (token_x_part[::-1].copy(),    # N max_lookback
                y_true[::-1].copy(),         # future, C
                token_y_part[::-1].copy(),   # N
                channel_label[::-1].copy(),  # N
                position_label[::-1].copy(), # N
                source_label[::-1].copy(),   # N
                tag_multihot[::-1].copy(),   # N max_tag_vocab_size
                np.array(list(y_true.shape)),   # output shape
                self.task_id_mapping["statistics"]  # task label
                )    
    
    def _tokenize_ts_transformation(self, x, y=None, target_series=None, method="differencing"):
        """
        training -
        x: list of (C, L) [np.array(C, L), np.array(C, L), ..., ], N C L
        not for inference
        """
        if self.force_resize_time_series_to_size_limit:
            for i in range(len(x)):
                x[i] = self.resize_time_series(x[i], target_length=self.lookback)
                if y is not None:
                    y[i] = self.resize_time_series(y[i], target_length=self.lookback)
            target_series = self.resize_time_series(target_series, target_length=self.lookback) if target_series is not None else None
        else:
            # check the length of the input
            len_limit = min(self.lookback, self.future)
            for i in range(len(x)):
                x[i] = x[i][:, -len_limit:] 
                if y is not None:
                    y[i] = y[i][:, -len_limit:]
            target_series = target_series[:, -len_limit:] if target_series is not None else None
        
        if y is None:
            if method == "decomposition":
                y = [time_series_transformation(xi, method=method) for xi in x]
                for index in range(len(x)):
                    x[index] = np.repeat(x[index], 3, axis=0)    # explode input for 3 decomposed result series
            else:
                y = [time_series_transformation(xi, method=method) for xi in x]

        token_x_part, y_true, token_y_part, channel_label, position_label, source_label, tag_multihot, _, _ = self._tokenize_imitation(x, y, target_series=target_series, do_resize=False)
        
        return (token_x_part[::-1].copy(),    # N max_lookback
                y_true[::-1].copy(),         # 
                token_y_part[::-1].copy(),   # N
                channel_label[::-1].copy(),  # N
                position_label[::-1].copy(), # N
                source_label[::-1].copy(),   # N
                tag_multihot[::-1].copy(),   # N max_tag_vocab_size
                np.array(list(y_true.shape)),   # output shape
                self.task_id_mapping[method]  # task label
                )
        
    @classmethod
    def time_series_to_tokens(cls, x, length=96, stride=4):
        # """
        # Transform x(C, L) to tokens(N C d)
        # length = lookback + future
        # stride: sampling step $m$
        # """
        # # Flip along the last axis
        # array_flipped = np.flip(x, axis=-1)
        # # Determine the number of slices
        # num_slices = (array_flipped.shape[-1] - length) // stride + 1
        # # Create the unfolded array
        # unfolded = np.lib.stride_tricks.as_strided(
        #     array_flipped,
        #     shape=array_flipped.shape[:-1] + (num_slices, length),
        #     strides=array_flipped.strides[:-1] + (array_flipped.strides[-1] * stride, array_flipped.strides[-1])
        # )
        # # Flip back along the last two axes
        # result = np.flip(unfolded, axis=-1)
        # result = np.flip(result, axis=-2).copy()      # (B) C N d
        # if len(result.shape) == 4:
        #     result = result.transpose(0, 2, 1, 3)  # (B) N C d
        # elif len(result.shape) == 3:
        #     result = result.transpose(1, 0, 2)     # N C d
        # return result
        """
        Transform x with shape (C, L) or (B, C, L) into tokens of shape (N, C, d) or (B, N, C, d).
        `length` specifies the lookback + future,
        `stride` specifies the sampling step.
        """
        if x.ndim == 2:  # Shape (C, L)
            C, L = x.shape
            num_tokens = (L - length) // stride + 1
            indices = np.arange(length)[None, :] + stride * np.arange(num_tokens)[:, None]
            tokens = x[:, indices]  # Result shape: (C, num_tokens, length)
            return tokens.transpose(1, 0, 2)  # Final shape: (num_tokens, C, length)
        elif x.ndim == 3:  # Shape (B, C, L)
            B, C, L = x.shape
            num_tokens = (L - length) // stride + 1
            indices = np.arange(length)[None, :] + stride * np.arange(num_tokens)[:, None]
            tokens = x[:, :, indices]  # Result shape: (B, C, num_tokens, length)
            return tokens.transpose(0, 2, 1, 3)  # Final shape: (B, num_tokens, C, length)
    
    @classmethod
    def get_valid_randomized_length(cls, L, x_max, y_max, n_max, factor=4, n_tries=8):
        x_min, x_max = 1, min(x_max, L-2)
        y_min, y_max = 1, min(y_max, (L-2) // 2)

        x_values = list(range(x_min, x_max + 1))
        y_values = list(range(y_min, y_max + 1))

        n_min_possible = L - x_max - 2 * y_max

        random.shuffle(x_values)
        random.shuffle(y_values)

        if n_min_possible < n_max:
            for x in x_values:
                for y in y_values:
                    n = L - x - 2 * y
                    if 1 <= n <= n_max:
                        return x, y, n
        
        current_factor = int(factor)
        for current_try in range(n_tries):
            for x in x_values:
                for y in y_values:
                    n = L - x - 2 * y
                    if 1 <= n <= n_max*current_factor:
                        warnings.warn(f"Possible information loss from token dropping - No valid randomized tokenization setting found with (L={L}, max_lookback={x_max}, max_future={y_max}, token_size_per_series_soft_threshold(soft_token_limit/C)={n_max}), relaxing the constraint n_samples <= token_size_per_series_soft_threshold... Infered lookback={x}, future={y}, token_size_per_series={n}, current_factor: {current_factor}, current_try: {current_try}", 
                                      category=UserWarning)
                        return x, y, n
                    
            current_factor = current_factor * factor
        
        raise ValueError(f"Randomized Training: No possible tokenization settings found with (L={L}, max_lookback={x_max}, max_future={y_max})")
        
    @classmethod
    def transform_ts_list_to_2d_array(cls, array_list, max_length_limit=65536):
        """
        array_list: list of L_ [np.array(), np.array(), ..., ]
        result: (N, L_max)
        """
        max_length = max(len(arr) for arr in array_list)

        result = np.zeros((len(array_list), max_length), dtype=array_list[0].dtype)

        for i, arr in enumerate(array_list):
            result[i, -len(arr):] = arr

        return result[:, -max_length_limit:]
    
    @classmethod
    def generate_random_mask(cls, L, p):
        """
        Randomly generate 1d mask with np.nan: [0, 0, 0, np.nan, 0, 0, 0]
        """
        mask = np.random.choice([0, np.nan], size=L, p=[1-p, p])
        return mask
    
    @classmethod
    def generate_continuous_mask(cls, L, p):
        mask = np.zeros(L, dtype=np.float32)
        num_nan = int(L * p)

        if num_nan == 0:
            return mask

        start_idx = np.random.randint(0, L - num_nan + 1)
        mask[start_idx:start_idx + num_nan] = np.nan

        return mask
    
    @classmethod
    def shuffle_array_along_axis(cls, a, axis):
        idx = np.arange(a.shape[axis])
        np.random.shuffle(idx)
        return np.take(a, idx, axis=axis)
    
    @classmethod
    def resize_time_series(cls, array, target_length, method='fft'):
        C, L = array.shape
        resized_array = np.empty((C, target_length))
        if method == 'fft':
            for i in range(C):
                resized_array[i] = resample(array[i], target_length)
        elif method == 'linear_interp':
            for i in range(C):
                x_old = np.linspace(0, 1, L)
                x_new = np.linspace(0, 1, target_length)
                resized_array[i] = np.interp(x_new, x_old, array[i])
        return resized_array

    @classmethod
    def shift_array(cls, arr, steps, axis, circular=True):
        if circular:
            arr = np.roll(arr, shift=steps, axis=axis)
            return arr
        else:
            result = np.empty_like(arr)
            if steps > 0:
                result.swapaxes(0, axis)[:steps] = 0
                result.swapaxes(0, axis)[steps:] = arr.swapaxes(0, axis)[:-steps]
            elif steps < 0:
                steps = -steps
                result.swapaxes(0, axis)[-steps:] = 0
                result.swapaxes(0, axis)[:-steps] = arr.swapaxes(0, axis)[steps:]
            else:
                result = arr
            return result