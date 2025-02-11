from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
from itertools import chain
import copy

import random

import re


def random_sample_dataset(lst, num_elements=16):
    if len(lst) < num_elements:
        return lst
    else:
        return random.sample(lst, num_elements)

def extract_data(s):
    pattern = re.compile(r'(?:\[(\d+\.\d+)(?:,(\d+))?(?:,(\d+))?\])?([^,]+)')
    ratios = []
    pred_lens = []
    batchsize = []
    filenames = []
    
    for match in pattern.findall(s):
        ratio = match[0]
        pred_len = match[1]
        batch_size = match[2]
        filename = match[3].strip()
        
        ratios.append(float(ratio) if ratio else 1.0)
        pred_lens.append(int(pred_len) if pred_len else 0)
        batchsize.append(int(batch_size) if batch_size else 0)
        filenames.append(filename)
    
    return ratios, pred_lens, batchsize, filenames

class RandomizedDataLoaderIter:
    def __init__(self, dataloaders, sample_len=2000):
        self.dataloaders = [iter(dl) for dl in dataloaders]
        self.active_iters = list(range(len(self.dataloaders)))
        self.sample_len = sample_len
        self.sample_counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_counter >= self.sample_len:
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
        return self.sample_len

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

def data_provider(args, flag, element_wise_shuffle=True):
    if ',' not in args.data_path:
        return data_provider_subset(args, flag)
    else:
        #data_names = args.data_path.split(',')
        data_few_shot_ratios, pred_lens, batchsizes, data_names = extract_data(args.data_path)
        pred_lens = [i  if i != 0 else args.pred_len for i in pred_lens]
        batchsizes = [i  if i != 0 else args.batch_size for i in batchsizes]
        
        mapping = [['custom' if 'ETT' not in dn else dn.split('.')[0], dn, r, bs, pdl] for dn, r, bs, pdl in zip(data_names, data_few_shot_ratios, batchsizes, pred_lens)]
        data_sets, data_loaders = [], []
        temp_args = copy.deepcopy(args)
        if flag in ['val', 'test']:
            for d in [mapping[-1]]:
                temp_args.data, temp_args.data_path, temp_args.few_shot_ratio, temp_args.batch_size, temp_args.batch_size_test, temp_args.pred_len = d[0], d[1], d[2], d[3], d[3], d[4]
                ds, dl = data_provider_subset(temp_args, flag)
                data_sets.append(ds)
                data_loaders.append(dl)
            return data_sets[-1], data_loaders[-1]
        
        current_datasets = random_sample_dataset(mapping[0:-1]) if args.transfer_learning else random_sample_dataset(mapping)
        current_datasets = current_datasets + [mapping[-2]]
        for d in current_datasets:
            temp_args.data, temp_args.data_path, temp_args.few_shot_ratio, temp_args.batch_size, temp_args.batch_size_test, temp_args.pred_len = d[0], d[1], d[2], d[3], d[3], d[4]
            ds, dl = data_provider_subset(temp_args, flag)
            data_sets.append(ds)
            data_loaders.append(dl)
        # For validation set and test set
        # For training set
        # if args.transfer_learning:
        #     return chain.from_iterable(data_sets[0:-1]), chain.from_iterable(data_loaders[0:-1]) if not element_wise_shuffle else RandomizedDataLoaderIter(data_loaders[0:-1])
        # else:
        return chain.from_iterable(data_sets), chain.from_iterable(data_loaders) if not element_wise_shuffle else RandomizedDataLoaderIter(data_loaders)

def data_provider_subset(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size if args.batch_size_test == 0 else args.batch_size_test
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        scale=args.scale,
        freq=freq,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        percent=int(getattr(args, 'few_shot_ratio', 1)*100),
        force_fair_comparison_for_extendable_and_extended_input_length=args.model == 'ICFormer' and flag == 'test'
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True)
    return data_set, data_loader
