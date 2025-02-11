from data_provider.data_factory import data_provider
from data_provider.ictsp_dataloader import build_icpretrain_dataloader, build_legacy_dataloader, TSPretrainDataset

from exp.exp_basic import Exp_Basic
from models.ICPretrain import ICPretrain
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.scientific_report import plot_aligned_heatmap, mts_visualize, vis_channel_forecasting, mts_visualize_horizontal

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import json
from collections import OrderedDict

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import logging

from types import SimpleNamespace

import gc

def create_directory(path):
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

class Exp_ICPretrain(Exp_Basic):
    def __init__(self, args):
        self.initialized = False
        super().__init__(args)
        self.writer = SummaryWriter('runs/{}_{}'.format(self.args.model_id, time.strftime("%Y%m%d-%H%M%S",time.localtime())))
        self.vali_times = 0
        self.test_times = 0
        self.steps = 0
        self.test_every = self.args.test_every
        self.early_stop = False
        self.additional_pred_resid_train_weight = 0
        self.current_best_rmse = float('inf')
        self.current_best_detailed_rmse = []
        self.current_best_detailed_rmse_original = []
        self.current_best_step = -1
        
        self.preds = None
        self.trues = None
        self.preds_vali = None
        self.trues_vali = None
        
        self.preds_best = None
        self.preds_vali_best = None

    def _build_model(self):
        with open(self.args.icpretrain_config_path, 'r') as file:
            config_data = json.load(file)
        self.icpretrain_configs = SimpleNamespace(**config_data)
        self.number_of_targets = getattr(self.icpretrain_configs, "number_of_targets", 0)
        
        model = ICPretrain(self.icpretrain_configs)#.float()
        
        if self.args.resume != 'none':
            state_dict = torch.load(self.args.resume, map_location='cpu')

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict, strict=False)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        if self.icpretrain_configs.compile:
            model = torch.compile(model)
        return model

    def _get_data(self, flag):
        if flag == 'train':
            if not self.icpretrain_configs.use_legacy_dataloader:
                data_set, data_loader = build_icpretrain_dataloader(self.icpretrain_configs, force_rebuild=not self.initialized)
                self.initialized = True
            else:
                ds, dl = data_provider(self.args, flag)
                data_set, data_loader = build_legacy_dataloader(ds, self.args, self.icpretrain_configs)
        if flag in ['val', 'test']:
            ds, dl = data_provider(self.args, flag)
            data_set, data_loader = build_legacy_dataloader(ds, self.args, self.icpretrain_configs)
        return data_set, data_loader

    def _select_optimizer(self):
        params = self.model.parameters()
        model_optim = optim.AdamW(params, lr=self.args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def check_memory(self):
        pass
    
    def preprocess_data(self, data):
        # flipped both in the tokenizer and the preprocessing here, just to ensure the "float to the right" alignment format properly applied on the channel dimension
        task_id = data[8].int().to(self.device, non_blocking=True)
        token_x_part = torch.nested.to_padded_tensor(data[0].float().to(self.device, non_blocking=True), 0).flip(1)
        y_true = torch.nested.to_padded_tensor(data[1].float().to(self.device, non_blocking=True), 0).flip(1) if task_id[0] != 1 else torch.nested.to_padded_tensor(data[1].int().to(self.device, non_blocking=True), 0).flip(1)      # C L or C
        token_y_part = torch.nested.to_padded_tensor(data[2].float().to(self.device, non_blocking=True), 0).flip(1) if task_id[0] != 1 else torch.nested.to_padded_tensor(data[2].int().to(self.device, non_blocking=True), 0).flip(1)
        channel_label = torch.nested.to_padded_tensor(data[3].int().to(self.device, non_blocking=True), 0).flip(1)
        position_label = torch.nested.to_padded_tensor(data[4].int().to(self.device, non_blocking=True), 0).flip(1)
        source_label = torch.nested.to_padded_tensor(data[5].int().to(self.device, non_blocking=True), 0).flip(1)
        tag_multihot = torch.nested.to_padded_tensor(data[6].float().to(self.device, non_blocking=True), 0).flip(1)
        y_true_shape = torch.nested.to_padded_tensor(data[7].int().to(self.device, non_blocking=True), 0)
        return task_id, token_x_part, y_true, token_y_part, channel_label, position_label, source_label, tag_multihot, y_true_shape

    def vali(self, vali_data, vali_loader, criterion, label='vali'):
        total_loss = []
        preds = []
        preds_add = []
        trues = []
        self.model.eval()
        cum_pred_flag = False
        print(f'Start Validation ({label})')
        with torch.no_grad():
            for i, data in enumerate(vali_loader):
                task_id, token_x_part, y_true, token_y_part, channel_label, position_label, source_label, tag_multihot, y_true_shape = self.preprocess_data(data)
                del data
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(token_x_part, token_y_part, channel_label, position_label, source_label, tag_multihot, y_true_shape, task_id)    # C L
                else:
                    outputs = self.model(token_x_part, token_y_part, channel_label, position_label, source_label, tag_multihot, y_true_shape, task_id)    # C L
                    
                if task_id[0] != 1:
                    # prune the output
                    n_max, ft_max = y_true_shape.max(dim=0)[0]
                    n_max, ft_max = n_max.item(), ft_max.item()
                    outputs = outputs[:, -n_max:, 0:ft_max]
                else:
                    # prune the output
                    n_max = y_true_shape.max(dim=0)[0]
                    n_max = n_max.item()
                    outputs = outputs[:, -n_max:, :]
                    
                outputs = outputs.permute(0, 2, 1)
                batch_y = y_true.permute(0, 2, 1)
                batch_x = token_x_part[:, -y_true.shape[1]:, :].permute(0, 2, 1)
                            
                add_loss_flag = False
                addtional_loss = 0
                # f_dim = -self.args.number_of_targets
                # if self.args.features == 'MS':
                #     f_dim = -1
                # outputs = outputs[:, -pred_len:, f_dim:]
                # batch_y = batch_y[:, -pred_len:, f_dim:]
                f_dim = 0
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                if cum_pred_flag:
                    for pred_timestep in range(cum_outputs.shape[1]):
                        if len(cum_preds_cache) <= pred_timestep+i:
                            cum_preds_cache.append([])
                        cum_preds_cache[pred_timestep+i].append(cum_outputs[:, [pred_timestep], f_dim:])
                    current_input_append = sum(cum_preds_cache[i]) / len(cum_preds_cache[i])
                    cum_input_x = torch.cat([cum_input_x[:, 1:, f_dim:], current_input_append], axis=1)
                    cum_outputs = cum_outputs[:, -pred_len:, f_dim:].detach().cpu()
                    cum_preds.append(cum_outputs.numpy())
                    
                preds.append(pred.float().numpy()[:, :, -self.number_of_targets:])
                trues.append(true.float().numpy()[:, :, -self.number_of_targets:])
                loss = criterion(pred, true)
                
                total_loss.append(loss)
        print(f'Validation ({label}): Inference Finished')
        # Averaging on the whole test set
        if cum_pred_flag:
            stacked_preds = np.concatenate(cum_preds, axis=0) # n, pred_len, C
        else:
            stacked_preds = np.concatenate(preds, axis=0) # n, pred_len, C
        
        if self.args.plot_full_details:
            stacked_trues = np.concatenate(trues, axis=0) # n, pred_len, C
            L_test = stacked_preds.shape[0] + stacked_preds.shape[1]
            channels = stacked_preds.shape[-1]
            cum_preds = np.zeros((L_test, channels))
            cum_trues = np.zeros((L_test, channels))
            vis_full_preds = np.ones((stacked_preds.shape[0], L_test, channels))*np.nan
            count = np.zeros((L_test, channels))

            print(f'Validation ({label}): Averaging Finished')

            for ci in range(stacked_preds.shape[0]):
                cum_preds[ci:ci+stacked_preds.shape[1]] += stacked_preds[ci]
                cum_trues[ci:ci+stacked_preds.shape[1]] += stacked_trues[ci]
                count[ci:ci+stacked_preds.shape[1]] += 1
                vis_full_preds[ci, ci:ci+stacked_preds.shape[1], :] = stacked_preds[ci]
            avg_preds = cum_preds / count
            avg_trues = cum_trues / count
            avg_trues_vis = np.concatenate([vali_data.data_pre[:, f_dim:], avg_trues], axis=0)
            avg_rmse = np.sqrt(np.nanmean((avg_preds - avg_trues)**2))
        
        print(f'Validation ({label}): Avg RMSE Finished')
        
        total_loss = np.average(total_loss)  
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        mae_ot, mse_ot, rmse_ot, mape_ot, mspe_ot, rse_ot, corr_ot = metric(preds[:, :, -1], trues[:, :, -1])
        if add_loss_flag:
            preds_add = np.concatenate(preds_add, axis=0)
            preds_add = preds_add.reshape(-1, preds_add.shape[-2], preds_add.shape[-1])
            mae_add, mse_add, rmse_add, mape_add, mspe_add, rse_add, corr_add = metric(preds_add, trues)
        total_loss = mse
        
        if label == 'test':
            print(f'Validation ({label}): Visualization')
            self.trues = trues
            self.preds = preds
            if add_loss_flag:
                try:
                    fig = plot_aligned_heatmap(ortho_matrix.squeeze().cpu().detach().numpy())
                    self.writer.add_figure(f'LossFigOrtho[0]', fig, self.test_times)
                    plt.clf()
                    fig = plot_aligned_heatmap(feature_resid[0].squeeze().cpu().detach().numpy())
                    self.writer.add_figure(f'LossFeatureResid[0]', fig, self.test_times)
                    plt.clf()
                    fig = plot_aligned_heatmap(feature_resid[1].squeeze().cpu().detach().numpy())
                    self.writer.add_figure(f'LossFeatureResid[-1]', fig, self.test_times)
                    plt.clf()
                    fig = plot_aligned_heatmap(pred_resid[0].squeeze().cpu().detach().numpy())
                    self.writer.add_figure(f'LossFigPredResid[0]', fig, self.test_times)
                    plt.clf()
                    fig = plot_aligned_heatmap(pred_resid[-1].squeeze().cpu().detach().numpy())
                    self.writer.add_figure(f'LossFigPredResid[-1]', fig, self.test_times)
                    plt.clf()
                except Exception as e:
                    print('Additional Loss Not Found: ', e)
            self.writer.add_scalar(f'Loss/{label}LossAvg', float(total_loss), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMSEAvg', float(mse), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMAEAvg', float(mae), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossRMSEAvg', float(rmse), self.test_times)
            self.writer.add_scalar(f'Loss/{label}OTLossMSEAvg', float(mse_ot), self.test_times)
            self.writer.add_scalar(f'Loss/{label}OTLossMAEAvg', float(mae_ot), self.test_times)
            self.writer.add_scalar(f'Loss/{label}OTLossRMSEAvg', float(rmse_ot), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMAPEAvg', float(mape), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMSPEAvg', float(mspe), self.test_times)
            if add_loss_flag:
                self.writer.add_scalar(f'Loss/{label}AddLossMSEAvg', float(mse_add), self.test_times)
                self.writer.add_scalar(f'Loss/{label}[AddLoss-Loss]MSEAvg', float(mse_add-mse), self.test_times)
                self.writer.add_scalar(f'Loss/{label}AddLossMAEAvg', float(mae_add), self.test_times)
            pred = pred.float().numpy()
            cbatch_x = torch.cat([batch_x[:, :, f_dim:], batch_y], dim=1).detach().cpu()
            cbatch_x = cbatch_x.float().numpy()
            met = f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, MSPE: {mspe:.4f}'
            #if cum_pred_flag:
            if self.args.plot_full_details:
                detailed_avg_rmse = []
                detailed_rmse = []
                for idx in range(avg_preds.shape[-1]):
                    detailed_avg_rmse.append(np.sqrt(np.nanmean((avg_preds[:, idx] - avg_trues[:, idx])**2)))
                    detailed_rmse.append(np.sqrt(np.nanmean((preds[:, :, idx] - trues[:, :, idx])**2)))
                self.current_best_rmse = min(avg_rmse, self.current_best_rmse)
                if avg_rmse == self.current_best_rmse:
                    self.current_best_step = self.test_times
                    self.current_best_detailed_rmse = detailed_avg_rmse
                    self.current_best_detailed_rmse_original = detailed_rmse
                    self.preds_best = self.preds.copy()
                    self.preds_vali_best = self.preds_vali.copy()
                title = f'Step: {self.test_times}, RMSE: {avg_rmse}, Best RMSE: {self.current_best_rmse}, Best Step: {self.current_best_step}, RMSEs: {detailed_avg_rmse}'
                self.writer.add_scalar(f'Loss/{label}LossRMSERecursive', float(avg_rmse), self.test_times)
                self.writer.add_scalar(f'Loss/{label}LossRMSERecursiveOT', float(detailed_avg_rmse[-1]), self.test_times)
                #self.writer.add_scalars('Loss/{label}LossRMSERecursiveDetailed', {str(index):value for index, value in enumerate(detailed_avg_rmse)}, self.test_times)
                print('Best Detailed Smoothed RMSE: ', self.current_best_detailed_rmse)
                print('Current Detailed Smoothed RMSE', detailed_avg_rmse)
                print('Current Original RMSE', detailed_rmse)
            if self.test_times % self.args.plot_every == 0:
                fig = mts_visualize(pred[0, :, -225:], cbatch_x[0, :, -225:], split_step=batch_x.shape[1], title=met, dpi=72, col_names=vali_data.col_names)
                if not os.path.exists("imgs"): os.makedirs("imgs")
                if not os.path.exists(f"imgs/{self.args.model_id}"): os.makedirs(f"imgs/{self.args.model_id}")
                fig.savefig(f"imgs/{self.args.model_id}/{self.test_times}.pdf", format="pdf", bbox_inches = 'tight')
                self.writer.add_figure('MTS_VS[1]', fig, self.test_times)
                plt.clf()
                #if cum_pred_flag:
                
                if not os.path.exists("imgs_testset"): os.makedirs("imgs_testset")
                if not os.path.exists(f"imgs_testset/{self.args.model_id}"): os.makedirs(f"imgs_testset/{self.args.model_id}")
                
                if self.args.plot_full_details:
                    fig = mts_visualize_horizontal(avg_preds, avg_trues_vis, split_step=avg_trues_vis.shape[0]-avg_preds.shape[0], title=title, dpi=120, width=50, col_names=vali_data.col_names)
                    fig.savefig(f"imgs_testset/{self.args.model_id}/{self.test_times}.pdf", format="pdf", bbox_inches = 'tight')
                    plt.clf()
                    fig = mts_visualize_horizontal(vis_full_preds[:, :, -1].T, np.repeat(avg_trues_vis[:, [-1]], vis_full_preds.shape[0], axis=1), split_step=avg_trues_vis.shape[0]-avg_preds.shape[0], title='OT Visualization', dpi=120, width=50)
                    fig.savefig(f"imgs_testset/{self.args.model_id}/{self.test_times}_OT.pdf", format="pdf", bbox_inches = 'tight')
                    plt.clf()
            if 'TAF' in self.args.model:
                if self.test_times % self.args.plot_every == 0:
                    vis_true = torch.cat([batch_x[:, :, f_dim:], batch_y], dim=1)
                    if 'mid' in masks:
                        current_mask = masks['mid'][0]
                    else:
                        current_mask = torch.ones(batch_x.shape[1], x_predictor_snapshot.shape[-1])
                    if not cum_pred_flag:
                        fig = vis_channel_forecasting(batch_x.shape[1], x_predictor_snapshot[0][:, -256:], outputs[0][:, -256:], vis_true[0][:, -256:], current_mask[:, -256:], col_names=vali_data.col_names)
                    else:
                        fig = vis_channel_forecasting(batch_x.shape[1], x_predictor_snapshot[0][:, -256:], cum_outputs.to(x_predictor_snapshot.device)[0][:, -256:], vis_true[0][:, -256:], current_mask[:, -256:], col_names=vali_data.col_names)
                    if x_predictor_snapshot.shape[-1] < 128:
                        self.writer.add_figure(f'ChannelVis', fig, self.test_times)
                    else:
                        if not os.path.exists("imgs"): os.makedirs("imgs")
                        if not os.path.exists(f"imgs/{self.args.model_id}"): os.makedirs(f"imgs/{self.args.model_id}")
                        fig.savefig(f"imgs/{self.args.model_id}/C_{self.test_times}.pdf", format="pdf", bbox_inches = 'tight')
                    plt.clf()
            self.test_times += 1
        if label == 'vali':
            self.trues_vali = trues
            self.preds_vali = preds
            if add_loss_flag:
                self.additional_pred_resid_train_weight = (mse - mse_add) / (mse+1e-5) if mse > mse_add else 0
            self.writer.add_scalar(f'Loss/{label}LossAvg', float(total_loss), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMSEAvg', float(mse), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMAEAvg', float(mae), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossRMSEAvg', float(rmse), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMAPEAvg', float(mape), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMSPEAvg', float(mspe), self.vali_times)
            self.vali_times += 1
        
        self.model.train()
        print('Validation Finished')
        gc.collect()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, configs=self.args)

        model_optim = self._select_optimizer()
        criterion = nn.MSELoss()
        criterion_cls = nn.CrossEntropyLoss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            pct_start = 0.002,
                                            div_factor = 1000,
                                            anneal_strategy='linear',
                                            epochs=self.args.train_epochs+1,
                                            steps_per_epoch=self.args.test_every,
                                            max_lr = self.args.learning_rate)
        epoch_time = time.time()
        for epoch in range(self.args.train_epochs):
            print(f'Starting Training Epoch: {epoch}')
            iter_count = 0
            train_loss = []
            self.model.train()
            for i, data in enumerate(train_loader):
                self.steps += 1
                iter_count += 1
                model_optim.zero_grad()
                
                task_id, token_x_part, y_true, token_y_part, channel_label, position_label, source_label, tag_multihot, y_true_shape = self.preprocess_data(data)
                del data
                # gc.collect()
                
                if torch.rand(1) < 0.05:
                    print(f'task_id: {task_id}, tkxp: {token_x_part.shape}, tkyp: {token_y_part.shape}, yts: {y_true.shape}')
                
                #print(f'Step {i}: {token_x_part.shape}, {y_true.shape}, {token_y_part.shape}, {channel_label.shape}, {position_label.shape}, {source_label.shape}, {tag_multihot.shape}, {y_true_shape}, {task_id}')

                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(token_x_part, token_y_part, channel_label, position_label, source_label, tag_multihot, y_true_shape, task_id)    # C L
                else:
                    outputs = self.model(token_x_part, token_y_part, channel_label, position_label, source_label, tag_multihot, y_true_shape, task_id)    # C L

                if self.model.tokenization_mode == 'sequential':
                    outputs, additional_loss = outputs
                else:
                    additional_loss = 0
                    
                if task_id[0] != 1:
                    # prune the output
                    n_max, ft_max = y_true_shape.max(dim=0)[0]
                    n_max, ft_max = n_max.item(), ft_max.item()
                    outputs = outputs[:, -n_max:, 0:ft_max]
                else:
                    # prune the output
                    n_max = y_true_shape.max(dim=0)[0]
                    n_max = n_max.item()
                    outputs = outputs[:, -n_max:, :]
                
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        if task_id[0] != 1:
                            if self.icpretrain_configs.use_legacy_dataloader:
                                loss = criterion(outputs[:, :, -self.number_of_targets:], y_true[:, :, -self.number_of_targets:])
                            else:
                                loss = additional_loss # criterion(outputs[:, :, -self.number_of_targets:], y_true[:, :, -self.number_of_targets:]) + additional_loss   # MSE
                            self.writer.add_scalar(f'Loss/TrainLossREG', float(loss.item()), self.steps)
                            self.writer.add_scalar(f'Loss/AddLossREG', float(additional_loss.item()), self.steps)
                        else:
                            if self.icpretrain_configs.use_legacy_dataloader:
                                loss = criterion(outputs[:, :, -self.number_of_targets:], y_true[:, :, -self.number_of_targets:])
                            else:
                                loss = additional_loss # criterion_cls(outputs.reshape(outputs.shape[0]*outputs.shape[1], -1), y_true.reshape(-1)) + additional_loss  # crossentropy
                            self.writer.add_scalar(f'Loss/TrainLossCLS', float(loss.item()), self.steps)
                            self.writer.add_scalar(f'Loss/AddLossCLS', float(additional_loss.item()), self.steps)
                else:
                    if task_id[0] != 1:
                        if self.icpretrain_configs.use_legacy_dataloader:
                            loss = criterion(outputs[:, :, -self.number_of_targets:], y_true[:, :, -self.number_of_targets:])
                        else:
                            loss = additional_loss # criterion(outputs[:, :, -self.number_of_targets:], y_true[:, :, -self.number_of_targets:]) + additional_loss  # MSE
                        self.writer.add_scalar(f'Loss/TrainLossREG', float(loss.item()), self.steps)
                        self.writer.add_scalar(f'Loss/AddLossREG', float(additional_loss.item()), self.steps)
                    else:
                        if self.icpretrain_configs.use_legacy_dataloader:
                            loss = criterion(outputs[:, :, -self.number_of_targets:], y_true[:, :, -self.number_of_targets:])
                        else:
                            loss = additional_loss # criterion_cls(outputs.reshape(outputs.shape[0]*outputs.shape[1], -1), y_true.reshape(-1)) + additional_loss  # crossentropy
                        self.writer.add_scalar(f'Loss/TrainLossCLS', float(loss.item()), self.steps)
                        self.writer.add_scalar(f'Loss/AddLossCLS', float(additional_loss.item()), self.steps)

                train_loss.append(loss.item())

                if (i + 1) % 50 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    
                    self.check_memory()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    if self.args.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    if (iter_count + 1) % self.args.gradient_accumulation == 0:
                        scaler.step(model_optim)
                        scaler.update()
                        model_optim.zero_grad()
                        
                else:
                    loss.backward()
                    if self.args.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    if (iter_count + 1) % self.args.gradient_accumulation == 0:
                        model_optim.step()
                        model_optim.zero_grad()    
                
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                if self.steps % self.test_every == 0:
                    model_optim.zero_grad()
                    print("Test Steps: {} cost time: {}".format(self.test_every, time.time() - epoch_time))
                    self.writer.add_scalar(f'LR/LearningRate', float(scheduler.get_last_lr()[0]), self.vali_times)
                    tl = np.average(train_loss)
                    vali_loss = self.vali(vali_data, vali_loader, criterion)
                    gc.collect()
                    print('Validation Finished (Vali)')
                    test_loss = self.vali(test_data, test_loader, criterion, label='test')
                    gc.collect()
                    print('Validation Finished (Test)')
                    
                    if self.test_times % self.args.plot_every == 0:
                        torch.save(getattr(self.model, '_orig_mod', self.model).state_dict(), f'pt_model_{self.args.seq_len}_{self.args.pred_len}_current.pth')
                    
                    print(model_optim)

                    print("Test Steps: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        self.steps, train_steps, tl, vali_loss, test_loss))
                    early_stopping(vali_loss, self.model, path)
                    print("[finished testing]")
                    if early_stopping.early_stop:
                        print("Early stopping")
                        self.early_stop = True
                    
            
            model_optim.zero_grad()
            
            if self.steps*self.icpretrain_configs.batch_size > self.icpretrain_configs.rebuild_every:
                print('Rebuilding Data Sources to Shuffle the Datapoints')
                del train_data, train_loader
                gc.collect()
                train_data, train_loader = self._get_data(flag='train')
                TSPretrainDataset.prepare_data_sources(self.icpretrain_configs, force_rebuild=True)
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            if self.early_stop:
                break

        best_model_path = path + '/' + 'checkpoint.pth'

        if self.args.resume != 'none':
            state_dict = torch.load(best_model_path, map_location='cpu')

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
                new_state_dict[name] = v
                
            self.model.load_state_dict(new_state_dict, strict=False)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                batch_x = data[0].float().to(self.device)
                batch_y = data[1].float().to(self.device)
                batch_x_mark = data[2].float().to(self.device)
                batch_y_mark = data[3].float().to(self.device)
                
                pred_len = batch_y.shape[1] - batch_x.shape[1]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        if 'TAF' in self.args.model:
                            outputs, masks, x_predictor_snapshot = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_mask=True)
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TadaT' in self.args.model or 'TSP' in self.args.model:
                            random_mask_rate = 1
                            rand_mask = torch.cat([torch.zeros(1, batch_x.shape[1], 1, device=batch_x.device).bool(), 
                                                   torch.rand(1, self.args.pred_len, 1, device=batch_x.device) < random_mask_rate
                                                  ], dim=1)
                            x = torch.cat([batch_x, torch.zeros_like(batch_y[:, -pred_len:, :], device=batch_x.device)], dim=1).masked_fill(rand_mask, 0)
                            x_mark = torch.cat([batch_x_mark, batch_y_mark[:, -pred_len:, :]], dim=1)
                            outputs = self.model(x, x_mark=x_mark, rand_mask=rand_mask, train=False)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'TAF' in self.args.model:
                        outputs, masks, x_predictor_snapshot = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_mask=True)
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'TadaT' in self.args.model or 'TSP' in self.args.model:
                        random_mask_rate = 1
                        rand_mask = torch.cat([torch.zeros(1, batch_x.shape[1], 1, device=batch_x.device).bool(), 
                                               torch.rand(1, self.args.pred_len, 1, device=batch_x.device) < random_mask_rate
                                              ], dim=1)
                        x = torch.cat([batch_x, torch.zeros_like(batch_y[:, -pred_len:, :], device=batch_x.device)], dim=1).masked_fill(rand_mask, 0)
                        x_mark = torch.cat([batch_x_mark, batch_y_mark[:, -pred_len:, :]], dim=1)
                        outputs = self.model(x, x_mark=x_mark, rand_mask=rand_mask, train=False)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -self.args.number_of_targets
                if self.args.features == 'MS':
                    f_dim = -1
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)
                outputs = outputs.float().detach().cpu().numpy()
                batch_y = batch_y.float().detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.float().detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.float().detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        print(preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        #mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, AVG RMSE details:{}'.format(mae, mse, rmse, mape, mspe, rse, self.current_best_detailed_rmse))
        print('RMSE details: {}'.format(self.current_best_detailed_rmse_original))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, details:{}'.format(mae, mse, rmse, mape, mspe, rse, self.current_best_detailed_rmse))
        f.write('\n')
        f.write('RMSE details:{}'.format(self.current_best_detailed_rmse_original))
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(pred_loader):
                batch_x = data[0].float().to(self.device)
                batch_y = data[1].float()
                batch_x_mark = data[2].float().to(self.device)
                batch_y_mark = data[3].float().to(self.device)
                pred_len = batch_y.shape[1] - batch_x.shape[1]

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
    
        
