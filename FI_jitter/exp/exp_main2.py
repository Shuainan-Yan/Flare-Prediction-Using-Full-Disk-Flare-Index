#coding=utf-8
from data_provider.data_factory import data_provider
from .exp_basic import Exp_Basic
from models import Autoformer,iTransformer,PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, Class_Metric
import csv
import numpy as np


import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def write_csv(filename,data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for slice_5x1 in data:

            flattened_slice = slice_5x1.flatten()

            writer.writerow(flattened_slice)

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'Autoformer': Autoformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()

                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:

                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
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
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'best_model.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def day_mae(self,true_file,pred_file):

        predicted_df = pd.read_csv(pred_file,header=None)
        actual_df = pd.read_csv(true_file, header=None)

        mae_per_column, mse_per_column, rmse_per_column, mape_per_column = [],[],[],[]

        for i in range(predicted_df.shape[1]):
            mae, mse, rmse, mape, mspe = metric(actual_df.iloc[:, i], predicted_df.iloc[:, i])
            mae_per_column.append(mae)
            mse_per_column.append(mse)
            rmse_per_column.append(rmse)
            mape_per_column.append(mape)

        for i in range(len(mape_per_column)):
             print('Day {}：rmse:{}, mse:{}, mae:{}, mape:{}'.format(i+1, rmse_per_column[i], mse_per_column[i], mae_per_column[i], mape_per_column[i]))

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'best_model.pth')))

        preds = []
        trues = []
        all_pred_data,all_true_data=[],[]

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                ### ysn:
                temp = outputs[:, -self.args.pred_len:, 0:]

                y = batch_y[:, -self.args.pred_len:, 0:].to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                ### ysn:
                all_pred_data.append(temp.detach().cpu().numpy())
                all_true_data.append(y.detach().cpu().numpy())

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        all_pred_data=np.array(all_pred_data)
        all_true_data=np.array(all_true_data)

        print('test shape:', preds.shape, trues.shape,all_pred_data.shape,all_true_data.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        all_pred_data = all_pred_data.reshape(-1, all_pred_data.shape[-2], all_pred_data.shape[-1])
        all_true_data = all_true_data.reshape(-1, all_true_data.shape[-2], all_true_data.shape[-1])
        print('test shape:', preds.shape, trues.shape, all_pred_data.shape, all_true_data.shape)


        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('rmse:{}, mse:{}, mae:{}, mape:{}'.format(rmse, mse, mae, mape))
       

        data_reshaped = all_true_data.reshape(-1, all_true_data.shape[-1])
        new_trues = test_data.inverse_transform(data_reshaped)
        new_trues = new_trues.reshape(-1, all_true_data.shape[-2], all_true_data.shape[-1])
        print("new_trues.shape", new_trues.shape)
        new_trues = new_trues[:,:,-1:]
        print("new_trues.shape", new_trues.shape)


        if self.args.model=='Autoformer':
            all_pred_data = all_pred_data * np.ones((1, 1, 1, all_true_data.shape[-1]))
            print('expanded array',all_pred_data)
        data_reshaped = all_pred_data.reshape(-1, all_pred_data.shape[-1])
        new_preds = test_data.inverse_transform(data_reshaped)
        new_preds = new_preds.reshape(-1, all_pred_data.shape[-2], all_pred_data.shape[-1])
        print("new_preds.shape", new_preds.shape)
        new_preds = new_preds[:,:,-1:]
        print("new_preds.shape", new_preds.shape)

        new_true_file = "./CSV/"+self.args.model+"_"+self.args.DataType+"_Inverse_y_true_lookback" + str(self.args.seq_len) + "_horizon" + str(
            self.args.pred_len) + ".csv"
        new_pred_file = "./CSV/"+self.args.model+"_"+self.args.DataType+"_Inverse_y_pred_lookback" + str(self.args.seq_len) + "_horizon" + str(
            self.args.pred_len) + ".csv"
        write_csv(new_true_file, new_trues)
        write_csv(new_pred_file, new_preds)

        self.day_mae(new_true_file, new_pred_file)
        mae, mse, rmse, mape, mspe = metric(new_preds, new_trues)
        print('rmse:{}, mse:{}, mae:{}, mape:{}'.format(rmse, mse, mae, mape))

        return

