import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
import os
import glob
import h5py
import re
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import pandas as pd
from collections import Counter
import copy
import numpy as np
import random

import pyarrow as pa
import pyarrow.dataset as ds

import sqlite3

from memory_profiler import profile

from data_provider.ictsp_tokenizer import InContextTimeSeriesTokenizer
import gc
import traceback

import h5py
import hdf5plugin

def count_directory_levels(path):
    normalized_path = os.path.normpath(path)
    levels = normalized_path.split(os.sep)
    return len(levels)

def extract_labels(input_string):
    region_pattern = r'__[a-zA-Z0-9_\-]+__([a-zA-Z0-9_\-]+)__'
    region_match = re.search(region_pattern, input_string)
    
    if not region_match:
        return []

    second_region = region_match.group(1)
    label_pattern = r'([ntm][a-z0-9_]+)'
    labels = re.findall(label_pattern, second_region)
    
    return labels

def get_hdf5_keys(file_path):
    with h5py.File(file_path, 'r') as hdf_file:
        keys = list(hdf_file.keys())
    return keys

def process_h5_file(h5_file):
    h5_datasets = []
    current_keys = get_hdf5_keys(h5_file)
    for key in current_keys:
        categories = extract_labels(key)
        for c in categories:
            h5_datasets.append(('h5', h5_file, key, c))
    return h5_datasets

def extract_arrow_file_info(path):
    """
    Recursively extracts information from each Arrow file in the given path and outputs it as a pandas DataFrame.
    
    Parameters:
    - path: The directory path containing the Arrow files.
    
    Returns:
    - A pandas DataFrame with columns 'file_name' and 'id' representing the information.
    """
    data = []

    # Recursively iterate over each file in the specified directory
    for root, _, files in os.walk(path):
        for file_name in files:
            if file_name.endswith('.arrow'):
                full_path = os.path.join(root, file_name)

                # Open the Arrow file as a dataset using the full path
                dataset = ds.dataset(full_path, format='ipc')

                # Read the 'id' column from the dataset
                table = dataset.to_table(columns=['id'])
                ids = table['id'].to_pylist()

                # Collect each [file_name, id] pair with full path
                for id_value in ids:
                    data.append(['arrow', full_path, id_value, None])

    return data

def get_file_paths(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)
    csv_files = [('csv', i, None, None) for i in csv_files]
    
    h5_files = glob.glob(os.path.join(folder_path, '**', '*.h5'), recursive=True)
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_h5_file, h5_files)
    
    h5_datasets = [dataset for result in results for dataset in result]

    arrow_files = extract_arrow_file_info(folder_path)
    
    return csv_files + h5_datasets + arrow_files

def extract_random_subarray(array, N):
    C, L = array.shape
    
    if N > L:
        return array  # Return the whole array if N > L
    
    # Generate a random start point for the subarray
    start = np.random.randint(0, L - N + 1)
    end = start + N
    
    return array[:, start:end]

def get_table_name(data_source_name):
    return f"dataset_{data_source_name}"

def generate_sine_waves_CL(N, L, frequency_range=(1, 10), amplitude_range=(0.5, 5)):
    sine_waves = np.zeros((N, L), dtype=np.float32)
    x = np.linspace(0, 2*np.pi, L)
    for i in range(N):
        frequency = np.random.uniform(*frequency_range)
        amplitude = np.random.uniform(*amplitude_range)
        phase = np.random.uniform(0, 2 * np.pi)
        sine_waves[i, :] = amplitude * np.sin(frequency * x + phase)
    return sine_waves

# beware of the random seed!!!!

class TSPretrainDataset(Dataset):
    def __init__(self, configs):
        super().__init__()
        self.root_path = configs.root_path
        self.max_sampling_len = configs.max_L_I
        self.training_soft_series_limit = configs.training_soft_series_limit
        self.current_ds_type = configs.current_ds_type    # cls or reg
        
        self.tokenizer = InContextTimeSeriesTokenizer(configs)

        self.scale = getattr(configs, "scale", 0)
        self.force_training_set_split_rate = getattr(configs, "force_training_set_split_rate", None)
        self.force_max_number_of_series = getattr(configs, "force_max_number_of_series", 0)
        
        #self.prepare_data_sources(configs)
        self.db_path = configs.db_path
        
        self.source_weight = {"cls": configs.source_weight_cls, "reg": configs.source_weight_reg}[self.current_ds_type]
        self.source_weight = [(key, value) for key, value in self.source_weight.items()]
        self.source_weight_name = [i[0] for i in self.source_weight]
        self.source_weight_p = [i[1] for i in self.source_weight]
        self.disabled = sum(self.source_weight_p) == 0
        if not self.disabled:
            self.source_weight_p = [i/sum(self.source_weight_p) for i in self.source_weight_p]
            print('Data Sources: ', self.source_weight_name)
            print('Weights: ', self.source_weight_p)

            self.token_type_weight = {'cls': configs.token_type_weight_cls, 'reg': configs.token_type_weight_reg}[self.current_ds_type]
            self.token_type = list(self.token_type_weight.keys())
            self.token_weight = np.array(list(self.token_type_weight.values()))
            self.token_weight = self.token_weight / self.token_weight.sum()
            self.total_len = configs.ds_len
        else:
            self.total_len = 0
            
        self.row_num_cache = {}
            
    @classmethod
    def prepare_data_sources(cls, configs, force_rebuild=False):
        # check if the table has been constructed
        db_path = configs.db_path
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS operation_status (
                id INTEGER PRIMARY KEY,
                operation_name TEXT UNIQUE,
                is_completed INTEGER
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS table_metadata (
                id INTEGER PRIMARY KEY,
                table_name TEXT UNIQUE,
                num_samples INTEGER
            )
            """)

            operation_name = "prepare_source_table"
            cursor = conn.execute("""
            SELECT is_completed
            FROM operation_status
            WHERE operation_name = ?
            """, (operation_name,))

            result = cursor.fetchone()
            finished = result is not None and result[0] == 1

            if not finished or force_rebuild:

                dir_level_shifting = count_directory_levels(configs.root_path) - 1

                def get_second_level_directory(path):
                    normalized_path = os.path.normpath(path)
                    _, path_tail = os.path.splitdrive(normalized_path)
                    parts = path_tail.split(os.sep)
                    if len(parts) > 2 + dir_level_shifting:
                        return parts[1 + dir_level_shifting]
                    else:
                        return None

                print(f"Initializing Data Source Metadata - Target Path: {configs.db_path}")
                data_sources = pd.DataFrame(get_file_paths(configs.root_path))
                data_sources.columns = ['type', 'path', 'key', 'category']
                data_sources['source_dir'] = data_sources['path'].map(get_second_level_directory)
                data_sources = data_sources.sample(frac=1).reset_index(drop=True)
                category_counts = data_sources.groupby(['category', 'source_dir']).size().reset_index(name='num_of_samples')

                # dataset_source1, dataset_source2, category_counts_source1, category_counts_source2
                # dataset_source1
                # type	path	key	category	source_dir
                # ...
                # category_counts_source1
                # category	source_dir	num_of_samples
                # ...
                source_weight_dirs = list(set(list(configs.source_weight_cls.keys()) + list(configs.source_weight_reg.keys())))
                for source_name in source_weight_dirs:
                    current_sources = data_sources.query(f"source_dir == '{source_name}'")
                    current_num_rows = current_sources.shape[0]
                    current_sources["id"] = current_sources.index + 1
                    current_sources.to_sql(get_table_name(source_name), conn, if_exists='replace', index=False,
                                           dtype={"id": "INTEGER PRIMARY KEY",
                                                  "type": "TEXT",
                                                  "path": "TEXT",
                                                  "key": "TEXT",
                                                  "category": "TEXT",
                                                  "source_dir": "TEXT"})
                    table_name = get_table_name(source_name)
                    conn.execute(f'CREATE INDEX idx_category_{table_name} ON {table_name} (category)')
                    data_sources.drop(current_sources.index, inplace=True)
                    conn.execute("""
                    INSERT OR REPLACE INTO table_metadata (table_name, num_samples)
                    VALUES (?, ?)
                    """, (table_name, current_num_rows))
                    #data_sources.to_sql('dataset', conn, if_exists='replace', index=False)
                category_counts.to_sql('category_counts', conn, if_exists='replace', index=False)
                # conn.execute('CREATE INDEX idx_category ON dataset (category)')

                # mark as complete
                conn.execute("""
                INSERT OR REPLACE INTO operation_status (operation_name, is_completed)
                VALUES (?, 1)
                """, (operation_name,))

                conn.commit()

        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        current_type = np.random.choice(self.token_type, p=self.token_weight)
        if current_type == "forecasting":
            source_table = np.random.choice(self.source_weight_name, p=self.source_weight_p)
            selected_source = self.get_random_row(n=1, table=get_table_name(source_table))
            data = self.read_single_file_from_source(selected_source, random_slicing=True)  # (C L)
            res = self.tokenizer.tokenize(data, task=current_type)
            if idx % 5 == 0 or idx > self.total_len - 5:
                gc.collect()
            return res
        elif current_type == "classification":
            x, y = self.form_cls_training_data()
            res = self.tokenizer.tokenize(x, y, task=current_type)
            return res
        elif current_type == "imitation":
            raise NotImplementedError("Pure imitation mode for training is not implemented")
        else:
            n_sources = np.random.randint(2, 100)
            source_tables = np.random.choice(self.source_weight_name, n_sources, p=self.source_weight_p)
            source_table_counter = Counter(source_tables)
            selected_source = []
            for current_source_table in source_table_counter:
                n_sources = source_table_counter[current_source_table]
                selected_source.append(self.get_random_row(n=n_sources, table=get_table_name(current_source_table)))
            selected_source = pd.concat(selected_source, ignore_index=True)
            data_list = self.read_file_from_source(selected_source, random_slicing=True, channel_limit=self.training_soft_series_limit)  # [..., (C L), ...]
            res = self.tokenizer.tokenize(data_list, task=current_type)
            if idx % 5 == 0 or idx > self.total_len - 5:
                gc.collect()
            return res
    
    def query_data_source(self, query):
        with sqlite3.connect(self.db_path) as conn:
            rows = pd.read_sql_query(query, conn)
            return rows

        # def get_random_row(self, n=1, table=None):
        #     conn = sqlite3.connect(self.db_path)
        #     query = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {n}"
        #     random_row = pd.read_sql_query(query, conn)
        #     conn.close()
        #     return random_row

    def get_random_row(self, n=1, table=None):
        with sqlite3.connect(self.db_path) as conn:
            if table not in self.row_num_cache:
                count_query = f"SELECT num_samples FROM table_metadata where table_name = '{table}'"
                total_rows = pd.read_sql_query(count_query, conn).iloc[0, 0]
                self.row_num_cache[table] = total_rows
            else:
                total_rows = self.row_num_cache[table]
            #print(table, total_rows, n, total_rows - n)
            if n > total_rows:
                #print('warning1', total_rows, n)
                n = np.random.randint(1, total_rows+1)
            #print('warning2', total_rows, n)
            random_offsets = np.random.randint(0, total_rows-n+1)
            #random_row = pd.DataFrame()

            #for offset in random_offsets:
            #    #query = f"SELECT * FROM {table} LIMIT 1 OFFSET {offset}"
            query = f"SELECT * FROM {table} WHERE id >= {random_offsets} LIMIT {n}"
            #query = f"SELECT * FROM {table} LIMIT {n} OFFSET {random_offsets}"
            #random_row = pd.concat([random_row, pd.read_sql_query(query, conn)])
            random_row = pd.read_sql_query(query, conn)
            return random_row

    def get_random_classification_row(self, num_categories=10, min_n_samples=4, max_n_samples=8):
        with sqlite3.connect(self.db_path) as conn:

            # dataset_source1, dataset_source2, category_counts_source1, category_counts_source2
            # dataset_source1
            # type	path	key	category	source_dir
            # ...
            # category_counts_source1
            # category	source_dir	num_of_samples
            # ...

            query_select_valid_categories = """
            SELECT *
            FROM category_counts
            WHERE num_of_samples > ?
            """
            valid_categories_df = pd.read_sql_query(query_select_valid_categories, conn, params=(min_n_samples,))

            random_categories = valid_categories_df.sample(min(num_categories, valid_categories_df.shape[0]))

            all_samples = []

            for index in range(random_categories.shape[0]):
                n_samples = random.randint(min_n_samples, max_n_samples)
                category = random_categories['category'].iloc[index]
                table = get_table_name(random_categories['source_dir'].iloc[index])
                query_select_samples = f"""
                SELECT * 
                FROM {table}
                WHERE category = ?
                ORDER BY RANDOM()
                LIMIT ?
                """
                samples_df = pd.read_sql_query(query_select_samples, conn, params=(category, n_samples))
                all_samples.append(samples_df)

            result_df = pd.concat(all_samples, ignore_index=True)

            return result_df
    
    def form_cls_training_data(self):
        n_tokens = np.random.randint(2048, 4096)
        
        n_categories = np.random.randint(2, 32)
        
        min_n_samples = np.random.randint(4, 16)
        max_n_samples = np.random.randint(16, 64)
        
        # min datapoints: 2*4=8, max datapoints: 32*64=2048
        # min avg n series in one dp: 2048/2048 = 1
        # max avg n series in one dp: 4096 / 8 = 512
        # avg_fetch_series = (n_tokens // (n_categories*max_n_samples), n_tokens // (n_categories*min_n_samples))
        
        source_rows = self.get_random_classification_row(num_categories=n_categories, min_n_samples=min_n_samples, max_n_samples=max_n_samples)
        n_dps = source_rows.shape[0]
        series_subset_size = n_tokens // n_dps
        
        x = self.read_file_from_source(source_rows, random_slicing=False, channel_limit=n_tokens,
                                       n_series_subset=series_subset_size,
                                       random_series_subset=np.random.rand()>0.5)
        y = source_rows.category.tolist()[0:len(x)]
        return x, y
    
    def read_single_file_from_source(self, source_samples, random_slicing=True, channel_limit=None, n_series_subset=0, random_series_subset=False):
        data_type = source_samples.iloc[0]["type"]
        if data_type == "h5":
            arr = self.read_hdf5(source_samples.iloc[0]["path"], source_samples.iloc[0]["key"])
        elif data_type == "csv":
            arr = self.read_csv(source_samples.iloc[0]["path"])
        elif data_type == "arrow":
            arr = self.read_arrow(source_samples.iloc[0]["path"], source_samples.iloc[0]["key"])
        else:
            raise NotImplementedError(f"File format: {data_type} is not implemented.")

        if self.force_training_set_split_rate is not None:
            L = arr.shape[1]
            limit = int(L * self.force_training_set_split_rate)
            arr = arr[:, 0:limit]
        
        if self.scale:
            arr = (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-6)
            arr = arr.clip(-30, 30)
            
        if random_slicing:
            arr = extract_random_subarray(arr, np.random.randint(16, self.max_sampling_len))
            
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            arr = generate_sine_waves_CL(np.random.randint(2, 50), np.random.randint(1024, 4096))
            print(f'WARNING: Using Alternative Data Source, Failed to Parse Input Source: {source_samples}')
            
        return arr

    def read_file_from_source(self, source_samples, random_slicing=True, channel_limit=None, n_series_subset=0, random_series_subset=False):
        arrs = []
        total_channels = 0
        for i in range(source_samples.shape[0]):
            if channel_limit is not None:
                if total_channels > channel_limit:
                    break
            data_type = source_samples.iloc[i]["type"]
            if data_type == "h5":
                arr = self.read_hdf5(source_samples.iloc[i]["path"], source_samples.iloc[i]["key"])
            elif data_type == "csv":
                arr = self.read_csv(source_samples.iloc[i]["path"])
            elif data_type == "arrow":
                arr = self.read_arrow(source_samples.iloc[i]["path"], source_samples.iloc[i]["key"])
            else:
                raise NotImplementedError(f"File format: {data_type} is not implemented.")

            if self.force_training_set_split_rate is not None:
                L = arr.shape[1]
                limit = int(L * self.force_training_set_split_rate)
                arr = arr[:, 0:limit]

            if self.scale:
                arr = (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-6)
                arr = arr.clip(-30, 30)
            
            if n_series_subset > 0:
                if random_series_subset:
                    n_series_samples = n_series_subset if arr.shape[0] >= n_series_subset else arr.shape[0]
                    indices = np.random.choice(arr.shape[0], n_series_samples, replace=False)
                    arr = arr[indices, :]
                else:
                    arr = arr[0:n_series_subset]
            
            if random_slicing:
                arr = extract_random_subarray(arr, np.random.randint(16, self.max_sampling_len))
                
            if arr.shape[0] == 0 or arr.shape[1] == 0:
                arr = generate_sine_waves_CL(np.random.randint(2, 50), np.random.randint(1024, 4096))
                print(f'WARNING: Using Alternative Data Source, Failed to Parse Input Source: {source_samples}')
                
            arrs.append(arr)
            total_channels += arr.shape[0]
            
        return arrs

    def read_csv(self, path):
        df = pd.read_csv(path)
        df.drop(columns='date', inplace=True)
        data = df.to_numpy(dtype=np.float32).T # C L
        fdim = - self.force_max_number_of_series
        return data[fdim:]
    
    # def read_hdf5(self, path, key):
    #     df = pd.read_hdf(path, key)
    #     df.drop(columns='date', inplace=True)
    #     return df.to_numpy(dtype=np.float32).T   # C L
    
    def read_hdf5(self, path, key):
        try:
            with h5py.File(path, 'r') as h5file:
                data = np.array(h5file[key]['table'])
                if 'values_block_1' in data.dtype.names:
                    data = data['values_block_1']
                else:
                    data = data['values_block_0']
            data = data.astype(np.float32).T
            fdim = - self.force_max_number_of_series
            return data[fdim:]
        except Exception as e:
            print(f"{e} --- Key '{key}' not found in file '{path}'.")
            return np.array([])

    def read_arrow(self, file_name, target_id):
        """
        Reads the datapoint corresponding to a given ID from an Arrow file and returns it as a numpy array.
        
        Parameters:
        - file_name: The Arrow file name from which to read the datapoint.
        - target_id: The ID of the datapoint to retrieve.
        
        Returns:
        - A numpy array corresponding to the datapoint with the given ID.
        """
        # Create a dataset scanner to avoid loading the entire table into memory
        dataset = ds.dataset(file_name, format='ipc')
        
        # Create a scanner with a filter to find the row with the matching ID
        scanner = dataset.scanner(filter=ds.field('id') == target_id, columns=['multivariate_time_series'], use_threads=True, batch_size=1024)
    
        # Iterate over the batches
        for batch in scanner.to_batches():
            if len(batch) > 0:
                # Extract the multivariate time series data from the batch
                np_array = batch.column('multivariate_time_series')[0].as_py()#.to_pylist()[0]
                
                # Convert the list of lists to a numpy array
                fdim = - self.force_max_number_of_series
                np_array = np.array(np_array)
                return np_array[fdim:]
        
        # If no matching ID was found, raise an error
        raise ValueError(f"No datapoint found with ID: {target_id}")

class ForecastingDatasetWrapper(Dataset):
    def __init__(self, dataset, configs):
        super().__init__()
        self.ds = dataset
        configs = copy.deepcopy(configs)
        configs.stage = "inference"
        self.tokenizer = InContextTimeSeriesTokenizer(configs)
        self.force_legacy_lookback_for_inference = getattr(configs, "force_legacy_lookback_for_inference", None)
        
    def __len__(self):
        return self.ds.__len__()
    
    def __getitem__(self, idx):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.ds[idx]   # L_I, C   L_P, C
        token_x_part, y_true, token_y_part, channel_label, position_label, source_label, tag_multihot, y_true_shape, task_id = self.tokenizer.tokenize(seq_x.T, task="forecasting", force_lookback=self.force_legacy_lookback_for_inference, force_future=seq_y.shape[0])
        y_true_from_ds = seq_y.T[::-1].copy()
        res = (token_x_part, y_true_from_ds, token_y_part, 
                channel_label, position_label, source_label, tag_multihot, 
                np.array(list(y_true_from_ds.shape)), task_id)
        if idx % 5 == 0 or idx > self.ds.__len__() - 5:
            gc.collect()
        return res
    
### Dataloader
def nested_collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    
    if isinstance(elem, np.ndarray):
        tensor_batch = list(map(torch.Tensor, batch))
        return torch.nested.nested_tensor(tensor_batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.LongTensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, tuple):
        transposed = zip(*batch)
        return [nested_collate_fn(samples) for samples in transposed]
    elif isinstance(elem, list):
        transposed = zip(*batch)
        return [nested_collate_fn(samples) for samples in transposed]
    else:
        print(batch)
        raise TypeError(f"batch must contain tensors, numpy arrays or numbers; found {elem_type}")
        
class RandomizedDataLoaderIter:
    def __init__(self, dataloaders, sample_len=None):
        self.dataloaders = [iter(dl) for dl in dataloaders]
        self.active_iters = list(range(len(self.dataloaders)))
        self.sample_len = None
        self.sample_counter = 0
        
        self.total_len = len(self.active_iters)

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_len is not None and self.sample_counter >= self.sample_len:
            raise StopIteration
        
        while self.active_iters:
            choice = random.choice(self.active_iters)
            try:
                data = next(self.dataloaders[choice])
                self.sample_counter += 1
                return data
            except StopIteration:
                self.active_iters.remove(choice)
        
        raise StopIteration

    def __len__(self):
        return self.total_len

def build_icpretrain_dataloader(configs, force_rebuild=False):
    TSPretrainDataset.prepare_data_sources(configs, force_rebuild=force_rebuild)
    cls_configs = copy.deepcopy(configs)
    cls_configs.current_ds_type = "cls"
    ds_cls = TSPretrainDataset(cls_configs)
    reg_configs = copy.deepcopy(configs)
    reg_configs.current_ds_type = "reg"
    ds_reg = TSPretrainDataset(reg_configs)
    
    ds_list = [ds_cls, ds_reg]
    dl_list = []
    for ds in ds_list:
        if not ds.disabled:
            dl_list.append(
                DataLoader(ds,
                           batch_size=configs.batch_size,
                           shuffle=False,
                           num_workers=configs.num_workers,
                           drop_last=True,
                           pin_memory=True,
                           persistent_workers=False,
                           prefetch_factor=2,
                           collate_fn=nested_collate_fn)
            )
    return None, dl_list[0]

def build_legacy_dataloader(ds, args, configs):
    ds_wrapped = ForecastingDatasetWrapper(ds, configs)
    dl = DataLoader(
        ds_wrapped,
        batch_size=args.batch_size if args.batch_size_test == 0 else args.batch_size_test,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=False,
        prefetch_factor=1,
        collate_fn=nested_collate_fn)
    return ds, dl