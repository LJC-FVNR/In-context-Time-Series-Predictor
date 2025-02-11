import os
import torch
import numpy as np
from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.scientific_report import mts_visualize, mts_visualize_horizontal
#import pmdarima as pm
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd

def generate_dateseries(periods=30):
    start_date = '1971-01-01'
    date_series = pd.date_range(start=start_date, periods=periods, freq='D')
    return date_series

def statsforecast_autoarima(x_input, forecasting_horizon):
    date = generate_dateseries(periods=len(x_input))
    df = pd.DataFrame({'unique_id': np.ones_like(x_input), 'ds': date, 'y': x_input})
    sf = StatsForecast(
        models = [AutoARIMA()],
        freq='D',
        n_jobs=-1
    )
    sf.fit(df)
    res = sf.predict(h=forecasting_horizon)
    return res.AutoARIMA.to_numpy()

def create_directory(path):
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

class Exp_Stat(object):
    def __init__(self, args):
        self.args = args
        self.model = None
        self.f_dim = -self.args.number_of_targets
        self.pca = None
        self.inputs = []
        if self.args.features == 'MS':
            self.f_dim = -1

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def overall_calculation(self, visualization=True):
        preds = self.preds
        trues = self.trues
        preds_vali = self.preds_vali
        trues_vali = self.trues_vali
        self.generate_inputs()
        vali_data, vali_loader = self._get_data(flag='val')
        f_dim = self.f_dim
        # n, pred_len, C
        L_test = preds.shape[0] + preds.shape[1]
        channels = preds.shape[-1]
        cum_preds = np.zeros((L_test, channels))
        cum_trues = np.zeros((L_test, channels))
        vis_full_preds = np.ones((preds.shape[0], L_test, channels))*np.nan
        count = np.zeros((L_test, channels))
        for ci in range(preds.shape[0]):
            cum_preds[ci:ci+preds.shape[1]] += preds[ci]
            cum_trues[ci:ci+preds.shape[1]] += trues[ci]
            count[ci:ci+preds.shape[1]] += 1
            vis_full_preds[ci, ci:ci+preds.shape[1], :] = preds[ci]
        avg_preds = cum_preds / count
        avg_trues = cum_trues / count
        avg_trues_vis = np.concatenate([vali_data.data_pre[:, f_dim:], avg_trues], axis=0)
        
        self.detailed_avg_rmse = []
        self.detailed_rmse = []
        self.detailed_rmse_vali = []
        for idx in range(avg_preds.shape[-1]):
            self.detailed_avg_rmse.append(np.sqrt(np.nanmean((avg_preds[:, idx] - avg_trues[:, idx])**2)))
            self.detailed_rmse.append(np.sqrt(np.nanmean((preds[:, :, idx] - trues[:, :, idx])**2)))
            self.detailed_rmse_vali.append(np.sqrt(np.nanmean((preds_vali[:, :, idx] - trues_vali[:, :, idx])**2)))
        
        if visualization:
            if not os.path.exists("imgs_testset"): os.makedirs("imgs_testset")
            if not os.path.exists(f"imgs_testset/{self.args.model_id}"): os.makedirs(f"imgs_testset/{self.args.model_id}")
            plot_full_details = self.args.plot_full_details
            if plot_full_details:
                #fig = mts_visualize_horizontal([avg_preds]+[vis_full_preds[i] for i in range(vis_full_preds.shape[0])], avg_trues_vis, split_step=avg_trues_vis.shape[0]-avg_preds.shape[0], title=title, dpi=120, width=50)
                fig = mts_visualize_horizontal(avg_preds, avg_trues_vis, split_step=avg_trues_vis.shape[0]-avg_preds.shape[0], title=self.args.model_id, dpi=120, width=50, col_names=vali_data.col_names)
                fig.savefig(f"imgs_testset/{self.args.model_id}/{self.args.model_id}.pdf", format="pdf", bbox_inches = 'tight')
                plt.clf()
                
            inputs = self.inputs
            assert inputs.shape[0] == trues.shape[0] and inputs.shape[-1] == trues.shape[-1], 'Inputs Shape: {}, Trues Shape: {}'.format(inputs.shape, trues.shape)
            cbatch_x = np.concatenate([inputs, trues], axis=1)
            
            n_cuts = inputs.shape[0] // self.args.batch_size
            fig = mts_visualize(preds[n_cuts*self.args.batch_size, :, -225:], cbatch_x[n_cuts*self.args.batch_size, :, -225:], split_step=self.args.seq_len, title=self.args.model_id, dpi=72, col_names=vali_data.col_names)
            if not os.path.exists("imgs"): os.makedirs("imgs")
            if not os.path.exists(f"imgs/{self.args.model_id}"): os.makedirs(f"imgs/{self.args.model_id}")
            fig.savefig(f"imgs/{self.args.model_id}/{self.args.model_id}.pdf", format="pdf", bbox_inches = 'tight')
            plt.clf()
            

    def PCA_fitting(self, X_train_dataset):
        x_exogs = []
        data_y = []
        for x, y, x_mark, y_mark in X_train_dataset:
            x_exog = x.copy().reshape(-1)
            y = y.reshape(-1)
            if len(data_y) > 0 and y.shape[0] != data_y[-1].shape[0]:
                break
            data_y.append(y)
            x_exogs.append(x_exog)
            
        x_exogs = np.stack(x_exogs)
        self.pca = PCA(n_components=0.99)
        self.pca.fit(x_exogs)
        print('Exog PCA N Components: {}'.format(self.pca.n_components_))
                
    def organize_data(self, dataset, strategy='flatten', exog=True):
        f_dim = self.f_dim
        data_x = []
        data_y = []
        finished = 0

        for x, y, x_mark, y_mark in dataset:
            x_exog = x.copy().reshape(-1)
            if self.pca is not None:
                x_exog = self.pca.transform(x_exog.reshape(1, -1)).reshape(-1)
            x = x[:, f_dim:]
            y = y[:, f_dim:]
            if strategy == 'flatten':
                x = x.reshape(-1)
                y = y.reshape(-1)
                if exog:
                    x = np.concatenate([x_exog, x])
                if len(data_y) > 0 and y.shape[0] != data_y[-1].shape[0]:
                    break
                data_x.append(x)
                data_y.append(y)
            
            elif strategy == 'independent':
                for c in range(x.shape[-1]):
                    current_x = x[:, c]
                    current_y = y[:, c]
                    if exog:
                        current_x = np.concatenate([x_exog, current_x])
                    if len(data_y) > 0 and current_y.shape[0] != data_y[-1].shape[0]:
                        finished = 1
                        break
                    data_x.append(current_x)
                    data_y.append(current_y)
            if finished:
                break
        data_x = np.stack(data_x)
        data_y = np.stack(data_y)
            
        print(data_x.shape, data_y.shape)
        return data_x, data_y
    
    def reorganize_data(self, preds, trues, preds_vali=None, trues_vali=None, strategy='flatten'):
        if strategy == 'flatten':
            self.preds = preds.reshape(-1, self.args.pred_len, self.args.number_of_targets)
            self.trues = trues.reshape(-1, self.args.pred_len, self.args.number_of_targets)
            if preds_vali is not None and trues_vali is not None:
                self.preds_vali = preds_vali.reshape(-1, self.args.pred_len, self.args.number_of_targets)
                self.trues_vali = trues_vali.reshape(-1, self.args.pred_len, self.args.number_of_targets)
        elif strategy == 'independent':
            self.preds = preds.reshape(-1, self.args.number_of_targets, self.args.pred_len).transpose(0, 2, 1)
            self.trues = trues.reshape(-1, self.args.number_of_targets, self.args.pred_len).transpose(0, 2, 1)
            if preds_vali is not None and trues_vali is not None:
                self.preds_vali = preds_vali.reshape(-1, self.args.number_of_targets, self.args.pred_len).transpose(0, 2, 1)
                self.trues_vali = trues_vali.reshape(-1, self.args.number_of_targets, self.args.pred_len).transpose(0, 2, 1)

            
    def generate_inputs(self):
        test_data, test_loader = self._get_data(flag='test')
        X_test, y_test = self.organize_data(test_data, strategy='flatten', exog=False)
        self.inputs = X_test.reshape(-1, self.args.seq_len, self.args.number_of_targets)
    
    def vali(self):
        pass

    def train(self, setting):
        {'ARIMA': self.train_stat_arima,
         'SARIMAX': self.train_stat_sarimax,
         'Repeat': self.train_stat_repeat,
         'Mean': self.train_stat_mean,
         'ExponentialSmoothing': self.train_stat_es,
         'HoltWinters': self.train_stat_holtwinters,
         'RandomForestExog': self.train_stat_randomforest_exog,
         'RandomForest': self.train_stat_randomforest,
         'LinearRegressionMultiExog': self.train_stat_lr_multi_exog,
         'LinearRegressionMulti': self.train_stat_lr_multi,
         'LinearRegressionIndExog': self.train_stat_lr_ind_exog,
         'LinearRegressionInd': self.train_stat_lr_ind,
         'LASSORegression': self.train_stat_lasso,
         'LASSORegressionExog': self.train_stat_lasso_exog,
         'RidgeRegression': self.train_stat_ridge,
         'RidgeRegressionExog': self.train_stat_ridge_exog,
         'LightGBMExog': self.train_stat_lightgbm_exog,
         'LightGBM': self.train_stat_lightgbm,
         'Seasonality': self.train_stat_seasonality,
        }[self.args.model](setting)
        
    def train_stat_repeat(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        def calculation(dataset):
            preds = []
            trues = []
            f_dim = self.f_dim
            # x: (L_I, C), y: (L_P, C)
            for x, y, x_mark, y_mark in dataset:
                x = x[:, f_dim:]
                y = y[:, f_dim:]
                if len(trues) > 0 and y.shape[0] != trues[-1].shape[0]:
                    break
                x_pred = np.stack([x[-1] for i in range(y.shape[0])])
                preds.append(x_pred)
                trues.append(y)
            return preds, trues
        
        preds_test, trues_test = calculation(test_data)
        preds_vali, trues_vali = calculation(vali_data)
        
        self.preds = np.stack(preds_test)
        self.trues = np.stack(trues_test)
        self.preds_vali = np.stack(preds_vali)
        self.trues_vali = np.stack(trues_vali)
    
    def train_stat_mean(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        def calculation(dataset):
            preds = []
            trues = []
            f_dim = self.f_dim
            # x: (L_I, C), y: (L_P, C)
            for x, y, x_mark, y_mark in dataset:
                x = x[:, f_dim:]
                y = y[:, f_dim:]
                if len(trues) > 0 and y.shape[0] != trues[-1].shape[0]:
                    break
                x_pred = np.stack([x.mean(axis=0) for i in range(y.shape[0])])
                preds.append(x_pred)
                trues.append(y)
            preds = np.stack(preds)
            trues = np.stack(trues)
            return preds, trues
        
        preds_test, trues_test = calculation(test_data)
        preds_vali, trues_vali = calculation(vali_data)
        
        self.preds = np.stack(preds_test)
        self.trues = np.stack(trues_test)
        self.preds_vali = np.stack(preds_vali)
        self.trues_vali = np.stack(trues_vali)
        
    def train_stat_seasonality(self, setting, season=52):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        def calculation(dataset):
            preds = []
            trues = []
            f_dim = self.f_dim
            # x: (L_I, C), y: (L_P, C)
            for x, y, x_mark, y_mark in dataset:
                x = x[:, f_dim:]
                y = y[:, f_dim:]
                if len(trues) > 0 and y.shape[0] != trues[-1].shape[0]:
                    break
                L_I, C = x.shape 
                L_P, _ = y.shape
                S = season
                predictions = np.zeros((L_P, C))

                for t in range(L_P):
                    historical_indices = np.arange(t, L_I, S)
                    historical_indices2 = np.arange(t, L_I, S+1)
                    historical_indices3 = np.arange(t, L_I, S-1)
                    if len(historical_indices) > 0:
                        historical_data = x[historical_indices] 
                        historical_data2 = x[historical_indices2] 
                        historical_data3 = x[historical_indices3] 
                        predictions[t] = (np.nanmean(historical_data, axis=0) + np.nanmean(historical_data2, axis=0) + np.nanmean(historical_data3, axis=0)) / 3

                    else:
                        predictions[t] = np.nan 

                x_pred = predictions
                preds.append(x_pred)
                trues.append(y)
            preds = np.stack(preds)
            trues = np.stack(trues)
            return preds, trues
        
        preds_test, trues_test = calculation(test_data)
        preds_vali, trues_vali = calculation(vali_data)
        
        self.preds = np.stack(preds_test)
        self.trues = np.stack(trues_test)
        self.preds_vali = np.stack(preds_vali)
        self.trues_vali = np.stack(trues_vali)
            
    def train_stat_arima(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        def calculation(dataset):
            preds = []
            trues = []
            f_dim = self.f_dim
            # x: (L_I, C), y: (L_P, C)
            i = 0
            for x, y, x_mark, y_mark in dataset:
                x = x[:, f_dim:]
                y = y[:, f_dim:]
                if len(trues) > 0 and y.shape[0] != trues[-1].shape[0]:
                    break
                x_pred = np.zeros_like(y)
                for c in range(x.shape[-1]):
                    #model = pm.auto_arima(x[:, c], seasonal=False, n_jobs=8)
                    #x_pred[:, c] = model.predict(n_periods=y.shape[0])
                    x_pred[:, c] = statsforecast_autoarima(x[:, c], y.shape[0])
                preds.append(x_pred)
                trues.append(y)
                i += 1
                print(f'Step {i} Finished')
            return preds, trues
            
        preds_test, trues_test = calculation(test_data)
        preds_vali, trues_vali = calculation(vali_data)
        
        self.preds = np.stack(preds_test)
        self.trues = np.stack(trues_test)
        self.preds_vali = np.stack(preds_vali)
        self.trues_vali = np.stack(trues_vali)
        
    def train_stat_es(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        def calculation(dataset):
            preds = []
            trues = []
            f_dim = self.f_dim
            # x: (L_I, C), y: (L_P, C)
            i = 0
            for x, y, x_mark, y_mark in dataset:
                x = x[:, f_dim:]
                y = y[:, f_dim:]
                if len(trues) > 0 and y.shape[0] != trues[-1].shape[0]:
                    break
                x_pred = np.zeros_like(y)
                for c in range(x.shape[-1]):
                    model = sm.tsa.ExponentialSmoothing(x[:, c], trend="add").fit()
                    x_pred[:, c] = model.forecast(steps=y.shape[0])
                preds.append(x_pred)
                trues.append(y)
                i += 1
                print(f'Step {i} Finished')
            return preds, trues
        
        preds_test, trues_test = calculation(test_data)
        preds_vali, trues_vali = calculation(vali_data)
        
        self.preds = np.stack(preds_test)
        self.trues = np.stack(trues_test)
        self.preds_vali = np.stack(preds_vali)
        self.trues_vali = np.stack(trues_vali)
            
    def train_stat_holtwinters(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        def calculation(dataset):
            preds = []
            trues = []
            f_dim = self.f_dim
            # x: (L_I, C), y: (L_P, C)
            i = 0
            for x, y, x_mark, y_mark in dataset:
                x = x[:, f_dim:]
                y = y[:, f_dim:]
                if len(trues) > 0 and y.shape[0] != trues[-1].shape[0]:
                    break
                x_pred = np.zeros_like(y)
                for c in range(x.shape[-1]):
                    model = sm.tsa.ExponentialSmoothing(x[:, c], trend="add", 
                                                                 seasonal="add", 
                                                                 seasonal_periods=12,
                                                                 damped_trend=True).fit()
                    x_pred[:, c] = model.forecast(steps=y.shape[0])
                preds.append(x_pred)
                trues.append(y)
                i += 1
                print(f'Step {i} Finished')
            return preds, trues
    
        preds_test, trues_test = calculation(test_data)
        preds_vali, trues_vali = calculation(vali_data)
        
        self.preds = np.stack(preds_test)
        self.trues = np.stack(trues_test)
        self.preds_vali = np.stack(preds_vali)
        self.trues_vali = np.stack(trues_vali)
        
    def train_stat_sarimax(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        self.PCA_fitting(train_data)
        
        def calculation(dataset):
            preds = []
            trues = []
            f_dim = self.f_dim
            # x: (L_I, C), y: (L_P, C)
            i = 0
            for x, y, x_mark, y_mark in dataset:
                x_exog = x.copy()
                x = x[:, f_dim:]
                y = y[:, f_dim:]
                if self.pca is not None:
                    x_exog = self.pca.transform(x_exog.reshape(1, -1)).reshape(-1)

                if len(trues) > 0 and y.shape[0] != trues[-1].shape[0]:
                    break
                x_pred = np.zeros_like(y)
                for c in range(x.shape[-1]):
                    model = sm.tsa.SARIMAX(x[:, c], 

                               order=(1, 0, 0), 
                               seasonal_order=(3, 0, 3, 52),
                               enforce_stationarity=False, 
                               enforce_invertibility=False).fit()
                    x_pred[:, c] = model.forecast(steps=y.shape[0])
                preds.append(x_pred)
                trues.append(y)
                i += 1
                print(f'Step {i} Finished')
            return preds, trues
        
        preds_test, trues_test = calculation(test_data)
        preds_vali, trues_vali = calculation(vali_data)
        
        self.preds = np.stack(preds_test)
        self.trues = np.stack(trues_test)
        self.preds_vali = np.stack(preds_vali)
        self.trues_vali = np.stack(trues_vali)
    
    
    def train_stat_lightgbm_exog(self, setting, strategy='independent', exog=True):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if strategy == 'independent' and exog:
            self.PCA_fitting(train_data)
        preds = []
        preds_vali = []
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        L_P = y_train.shape[-1]
        params = {
                'boosting_type': 'gbdt', 
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9, 
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0 ,
                'early_stopping_rounds': 10
            }
        for l in range(L_P):
            train_data = lgb.Dataset(X_train, label=y_train[:, l])
            vali_data = lgb.Dataset(X_vali, label=y_vali[:, l], reference=train_data)
            # X_train: (N, D)
            # y_train: (N, c)
            
            gbm = lgb.train(params,
                    train_data,
                    num_boost_round=50, 
                    valid_sets=vali_data)
            y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
            y_pred_vali = gbm.predict(X_vali, num_iteration=gbm.best_iteration)
            preds.append(y_pred)
            preds_vali.append(y_pred_vali)
            print(f'Step {l} Finished')
        preds = np.stack(preds).T
        preds_vali = np.stack(preds_vali).T
        self.reorganize_data(preds, y_test, preds_vali, y_vali, strategy=strategy)
        
    def train_stat_lightgbm(self, setting, strategy='independent', exog=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if strategy == 'independent' and exog:
            self.PCA_fitting(train_data)
        preds = []
        preds_vali = []
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        L_P = y_train.shape[-1]
        params = {
                'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9, 
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0 ,
                'early_stopping_rounds': 10
            }
        for l in range(L_P):
            train_data = lgb.Dataset(X_train, label=y_train[:, l])
            vali_data = lgb.Dataset(X_vali, label=y_vali[:, l], reference=train_data)
            # X_train: (N, D)
            # y_train: (N, c)
            
            gbm = lgb.train(params,
                    train_data,
                    num_boost_round=50, 
                    valid_sets=vali_data)
            y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
            y_pred_vali = gbm.predict(X_vali, num_iteration=gbm.best_iteration)
            preds.append(y_pred)
            preds_vali.append(y_pred_vali)
            print(f'Step {l} Finished')
        preds = np.stack(preds).T
        preds_vali = np.stack(preds_vali).T
        self.reorganize_data(preds, y_test, preds_vali, y_vali, strategy=strategy)

    def train_stat_randomforest_exog(self, setting, strategy='independent', exog=True):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if strategy == 'independent' and exog:
            self.PCA_fitting(train_data)
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, verbose=1, n_jobs=8)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_vali = model.predict(X_vali)
        self.reorganize_data(y_pred, y_test, y_pred_vali, y_vali, strategy=strategy)

    def train_stat_randomforest(self, setting, strategy='independent', exog=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if strategy == 'independent' and exog:
            self.PCA_fitting(train_data)
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, verbose=1, n_jobs=8)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_vali = model.predict(X_vali)
        self.reorganize_data(y_pred, y_test, y_pred_vali, y_vali, strategy=strategy)

    ###########################################################################
    def train_stat_lr_multi_exog(self, setting, strategy='flatten', exog=True):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if exog:
            self.PCA_fitting(train_data)
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_vali = model.predict(X_vali)
        self.reorganize_data(y_pred, y_test, y_pred_vali, y_vali, strategy=strategy)

    def train_stat_lr_multi(self, setting, strategy='flatten', exog=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if exog:
            self.PCA_fitting(train_data)
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_vali = model.predict(X_vali)
        self.reorganize_data(y_pred, y_test, y_pred_vali, y_vali, strategy=strategy)

    def train_stat_lr_ind_exog(self, setting, strategy='independent', exog=True):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if exog:
            self.PCA_fitting(train_data)
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_vali = model.predict(X_vali)
        self.reorganize_data(y_pred, y_test, y_pred_vali, y_vali, strategy=strategy)

    def train_stat_lr_ind(self, setting, strategy='independent', exog=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if exog:
            self.PCA_fitting(train_data)
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_vali = model.predict(X_vali)
        self.reorganize_data(y_pred, y_test, y_pred_vali, y_vali, strategy=strategy)
    ###########################################################################

    def train_stat_lasso_exog(self, setting, strategy='independent', exog=True):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if exog:
            self.PCA_fitting(train_data)
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        model = Lasso()
        #model = GridSearchCV(model, {'alpha': np.logspace(-4, 4, 25)}, cv=5, verbose=2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_vali = model.predict(X_vali)
        self.reorganize_data(y_pred, y_test, y_pred_vali, y_vali, strategy=strategy)
        
    def train_stat_lasso(self, setting, strategy='independent', exog=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if exog:
            self.PCA_fitting(train_data)
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        model = Lasso()
        #model = GridSearchCV(model, {'alpha': np.logspace(-4, 4, 25)}, cv=5, verbose=2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_vali = model.predict(X_vali)
        self.reorganize_data(y_pred, y_test, y_pred_vali, y_vali, strategy=strategy)
        
    def train_stat_ridge_exog(self, setting, strategy='independent', exog=True):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if exog:
            self.PCA_fitting(train_data)
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        model = Ridge()
        #model = GridSearchCV(model, {'alpha': np.logspace(-4, 4, 25)}, cv=5, verbose=2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_vali = model.predict(X_vali)
        self.reorganize_data(y_pred, y_test, y_pred_vali, y_vali, strategy=strategy)
        
    def train_stat_ridge(self, setting, strategy='independent', exog=False):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        if exog:
            self.PCA_fitting(train_data)
        f_dim = self.f_dim
        X_train, y_train = self.organize_data(train_data, strategy=strategy, exog=exog)
        X_vali, y_vali = self.organize_data(vali_data, strategy=strategy, exog=exog)
        X_test, y_test = self.organize_data(test_data, strategy=strategy, exog=exog)
        model = Ridge()
        #model = GridSearchCV(model, {'alpha': np.logspace(-4, 4, 25)}, cv=5, verbose=2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_vali = model.predict(X_vali)
        self.reorganize_data(y_pred, y_test, y_pred_vali, y_vali, strategy=strategy)
        
    def test(self, setting):
        self.overall_calculation()
        data_name = self.args.data_path.split('.')[0]
        path = f'prediction/{data_name}'
        create_directory(path)
        np.savez(f'{path}/trues.npz', vali=self.trues_vali, test=self.trues)
        np.savez(f'{path}/{self.args.model}.npz', vali=self.preds_vali, test=self.preds)
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        torch.save(vali_data, f'{path}/vali_dataset.pth')
        torch.save(test_data, f'{path}/test_dataset.pth')
        mae, mse, rmse, mape, mspe, rse, corr = metric(self.preds, self.trues)
        mae_vali, mse_vali, rmse_vali, mape_vali, mspe_vali, rse_vali, corr_vali = metric(self.preds_vali, self.trues_vali)
        mae_ot, mse_ot, rmse_ot, mape_ot, mspe_ot, rse_ot, corr_ot = metric(self.preds[:, :, [-1]], self.trues[:, :, [-1]])
        print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, details:{}'.format(mae, mse, rmse, mape, mspe, rse, self.detailed_avg_rmse))
        print('RMSE details: {}'.format(self.detailed_rmse))
        print('mae_ot:{}, mse_ot:{}, rmse_ot:{}, mape_ot:{}, mspe_ot:{}, rse_ot:{}'.format(mae_ot, mse_ot, rmse_ot, mape_ot, mspe_ot, rse_ot))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, AVG RMSE details:{}'.format(mae, mse, rmse, mape, mspe, rse, self.detailed_avg_rmse))
        f.write('\n')
        f.write('RMSE details:{}'.format(self.detailed_rmse))
        f.write('\n')
        f.write('RMSE Vali details:{}'.format(self.detailed_rmse_vali))
        f.write('\n')
        f.close()
