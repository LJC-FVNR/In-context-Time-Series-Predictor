from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, TimesNet, TiDE, ICTSP
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.scientific_report import mts_visualize
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def create_directory(path):
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
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
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'TimesNet': TimesNet,
            'TiDE': TiDE,
            'ICTSP': ICTSP
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        if self.args.resume != 'none':
            model.load_state_dict(torch.load(self.args.resume))

        # if torch.cuda.is_available() and torch.__version__ > '2.0':
        model = torch.compile(model)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        params = self.model.parameters()
        model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, label='vali'):
        total_loss = []
        preds = []
        preds_add = []
        trues = []
        self.model.eval()
        print(f'Start Validation ({label})')
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                
                pred_len = batch_y.shape[1] - batch_x.shape[1]
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=True)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                f_dim = -self.args.number_of_targets
                if self.args.features == 'MS':
                    f_dim = -1
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                    
                preds.append(pred.numpy())
                trues.append(true.numpy())
                loss = criterion(pred, true)
                
                total_loss.append(loss)
        print(f'Validation ({label}): Inference Finished')
        
        print(f'Validation ({label}): Avg RMSE Finished')
        
        total_loss = np.average(total_loss)  
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        mae_ot, mse_ot, rmse_ot, mape_ot, mspe_ot, rse_ot, corr_ot = metric(preds[:, :, -1], trues[:, :, -1])
        total_loss = mse
        
        if label == 'test':
            print(f'Validation ({label}): Visualization')
            self.trues = trues
            self.preds = preds
            self.writer.add_scalar(f'Loss/{label}LossAvg', float(total_loss), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMSEAvg', float(mse), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMAEAvg', float(mae), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossRMSEAvg', float(rmse), self.test_times)
            self.writer.add_scalar(f'Loss/{label}OTLossMSEAvg', float(mse_ot), self.test_times)
            self.writer.add_scalar(f'Loss/{label}OTLossMAEAvg', float(mae_ot), self.test_times)
            self.writer.add_scalar(f'Loss/{label}OTLossRMSEAvg', float(rmse_ot), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMAPEAvg', float(mape), self.test_times)
            self.writer.add_scalar(f'Loss/{label}LossMSPEAvg', float(mspe), self.test_times)
            pred = pred.numpy()
            cbatch_x = torch.cat([batch_x[:, :, f_dim:], batch_y], dim=1).detach().cpu()
            cbatch_x = cbatch_x.numpy()
            met = f'MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, MSPE: {mspe:.4f}'
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
            self.test_times += 1
        if label == 'vali':
            self.trues_vali = trues
            self.preds_vali = preds
            self.writer.add_scalar(f'Loss/{label}LossAvg', float(total_loss), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMSEAvg', float(mse), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMAEAvg', float(mae), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossRMSEAvg', float(rmse), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMAPEAvg', float(mape), self.vali_times)
            self.writer.add_scalar(f'Loss/{label}LossMSPEAvg', float(mspe), self.vali_times)
            self.vali_times += 1
        
        self.model.train()
        print('Validation Finished')
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        initialized = False

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, configs=self.args)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            pct_start = 0.002,
                                            div_factor = 10,
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.steps += 1
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                pred_len = batch_y.shape[1] - batch_x.shape[1]
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=True)

                drop_mask = None
                if self.args.random_drop_training:
                    if torch.rand(1).item() > 0:
                        random_drop_rate = torch.rand(1).item()
                        drop_mask = torch.rand(1, 1, batch_x.shape[2], device=batch_x.device) < 1-random_drop_rate
                        batch_x = batch_x.masked_fill(drop_mask, 0)
                        batch_y = batch_y.masked_fill(drop_mask, 0)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)     
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) # , batch_y
                
                
                f_dim = -self.args.number_of_targets
                if self.args.features == 'MS':
                    f_dim = -1
                batch_y = batch_y[:, -pred_len:, f_dim:]
                
                outputs = outputs[:, -pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                self.writer.add_scalar(f'Loss/TrainLossTOT', float(loss.item()), self.steps)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

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
                    
                if self.args.model == 'ICTSP' and self.model.model.linear_warm_up_counter == self.model.model.linear_warmup_steps and not initialized:
                    initialized = True
                    model_optim = self._select_optimizer()
                    scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                        pct_start = 0.002,
                                                        div_factor = 10,
                                                        anneal_strategy='linear',
                                                        epochs=self.args.train_epochs+1,
                                                        steps_per_epoch=self.args.test_every,
                                                        max_lr = self.args.learning_rate)
                    self.args.gradient_accumulation = 8
                    self.args.batch_size = 4
                    break

                if self.steps % self.test_every == 0:
                    print("Test Steps: {} cost time: {}".format(self.test_every, time.time() - epoch_time))
                    self.writer.add_scalar(f'LR/LearningRate', float(scheduler.get_last_lr()[0]), self.vali_times)
                    tl = np.average(train_loss)
                    vali_loss = self.vali(vali_data, vali_loader, criterion)
                    print('Validation Finished (Vali)')
                    test_loss = self.vali(test_data, test_loader, criterion, label='test')
                    print('Validation Finished (Test)')
                    
                    print(model_optim)

                    print("Test Steps: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        self.steps, train_steps, tl, vali_loss, test_loss))
                    early_stopping(vali_loss, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        self.early_stop = True
            
            model_optim.zero_grad()
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            if self.early_stop:
                break
            # Refresh train_loader for multi-dataset training mode
            train_data, train_loader = self._get_data(flag='train')

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                pred_len = batch_y.shape[1] - batch_x.shape[1]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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

                f_dim = -self.args.number_of_targets
                if self.args.features == 'MS':
                    f_dim = -1
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
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
        # preds = np.array(preds)
        # trues = np.array(trues)
        # inputx = np.array(inputx)

        # preds = preds.transpose((0,2,1))#.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.transpose((0,2,1))#.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.transpose((0,2,1))#.reshape(-1, inputx.shape[-2], inputx.shape[-1])

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                pred_len = batch_y.shape[1] - batch_x.shape[1]

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
    
        
