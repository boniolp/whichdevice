import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Helpers.utils import Classifmetrics, NILMmetrics

    
class TSDataset(torch.utils.data.Dataset):
    """
    MAP-Style PyTorch Time series Dataset with possibility of scaling
    
    - X matrix of TS input, can be 2D or 3D, Dataframe instance or Numpy array instance.
    - Labels : y labels associated to time series for classification. Possible to be None.
    - scaler : provided type of scaler (sklearn StandardScaler, MinMaxScaler instance for example).
    - scale_dim : list of dimensions to be scaled in case of multivariate TS.
    """
    def __init__(self, X, labels=None, scaler=False, scale_dim=None):
        
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(labels, pd.core.frame.DataFrame):
            labels = labels.values
        
        if scaler:
            # ==== Multivariate case ==== #
            if len(X.shape)==3:
                self.scaler_list = []
                self.samples = X
                if scale_dim is None:                    
                    for i in range(X.shape[1]):
                        self.scaler_list.append(StandardScaler())
                        self.samples[:,i,:] = self.scaler_list[i].fit_transform(X[:,i,:].T).T.astype(np.float32)
                else:
                    for idsc, i in enumerate(scale_dim):
                        self.scaler_list.append(StandardScaler())
                        self.samples[:,i,:] = self.scaler_list[idsc].fit_transform(X[:,i,:].T).T.astype(np.float32)
                        
            # ==== Univariate case ==== #
            else:
                self.scaler_list = [StandardScaler()]
                self.samples = self.scaler_list[0].fit_transform(X.T).T.astype(np.float32)
        else:
            self.samples = X
            
        if len(self.samples.shape)==2:
            self.samples = np.expand_dims(self.samples, axis=1)
        
        if labels is not None:
            self.labels = labels.ravel()
            assert len(self.samples)==len(self.labels), f"Number of X sample {len(self.samples)} doesn't match number of y sample {len(self.labels)}."
        else:
            self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        if self.labels is None:
            return self.samples[idx]
        else:
            return self.samples[idx], self.labels[idx]


class TSDatasetScaling(torch.utils.data.Dataset):
    """
    MAP-Style PyTorch Time series Dataset
    
    Scaling computed on the fly
    """
    def __init__(self, X, labels=None, scale_data=False, inst_scaling=True,
                 st_date=None, mask_date=None, freq='30T', list_exo_variables=['hours, dow'], cosinbase=True, newRange=(-1, 1)):

        self.scale_data = scale_data
        self.inst_scaling = inst_scaling

        self.freq = freq
        self.list_exo_variables = list_exo_variables
        self.cosinbase = cosinbase
        self.newRange = newRange
        self.L = X.shape[-1]

        if st_date is not None:
            assert list_exo_variables is not None and len(list_exo_variables) > 0, "Please provide list of exo variable if st_date not None."
            assert self.freq is not None, "Variable freq not defined but st_date provided."

            self.st_date = st_date[mask_date].values.flatten()

            if self.cosinbase:
                self.n_var = 2*len(self.list_exo_variables)
            else:
                self.n_var = len(self.list_exo_variables)
        else:
            self.n_var = None
        
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(labels, pd.core.frame.DataFrame):
            labels = labels.values
        
        self.samples = X
            
        if len(self.samples.shape)==2:
            self.samples = np.expand_dims(self.samples, axis=1)

        if not inst_scaling:
            self.mean = np.squeeze(np.mean(self.samples, axis=(0, 2), keepdims=True), axis=0)
            self.std  = np.squeeze(np.std(self.samples, axis=(0, 2), keepdims=True) + 1e-9, axis=0)
        
        if labels is not None:
            self.labels = labels.ravel()
            assert len(self.samples)==len(self.labels), f"Number of X sample {len(self.samples)} doesn't match number of y sample {len(self.labels)}."
        else:
            self.labels = labels

    def _create_exogene(self, idx):
        
        np_extra = np.empty((self.n_var, self.L)).astype(np.float32)
        tmp = pd.date_range(start=self.st_date[idx], periods=self.L, freq=self.freq)
        
        k = 0
        for exo_var in self.list_exo_variables:
            if exo_var=='month':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.month.values/12.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.month.values/12.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.month.values, xmin=1, xmax=12, newRange=self.newRange)
                    k+=1
            elif exo_var=='dom':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.day.values/31.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.day.values/31.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.month.values, xmin=1, xmax=12, newRange=self.newRange)
                    k+=1
            elif exo_var=='dow':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.dayofweek.values/7.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.dayofweek.values/7.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.month.values, xmin=1, xmax=7, newRange=self.newRange)
                    k+=1
            elif exo_var=='hour':
                if self.cosinbase:
                    np_extra[k, :]   = np.sin(2 * np.pi * tmp.hour.values/24.0)
                    np_extra[k+1, :] = np.cos(2 * np.pi * tmp.hour.values/24.0)
                    k+=2
                else:
                    np_extra[k, :]   = self._normalize(tmp.month.values, xmin=0, xmax=24, newRange=self.newRange)
                    k+=1
            else:
                raise ValueError("Embedding unknown for these Data. Only 'month', 'dow', 'dom' and 'hour' supported, received {}"
                                    .format(exo_var))   

        return np_extra

    def _normalize(self, x, xmin, xmax, newRange): 
        if xmin is None:
            xmin = np.min(x)
        if xmax is None:
            xmax = np.max(x) 
            
        norm = (x - xmin)/(xmax - xmin) 
        if newRange == (0, 1):
            return norm 
        elif newRange != (0, 1):
            return norm * (newRange[1] - newRange[0]) + newRange[0] 

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        tmp_sample = self.samples[idx].copy()

        if self.scale_data:
            if self.inst_scaling:
                tmp_sample = (tmp_sample - np.mean(tmp_sample, axis=1, keepdims=True)) / (np.std(tmp_sample, axis=1, keepdims=True) + 1e-9)
            else:
                tmp_sample = (tmp_sample - self.mean) / self.std

        if self.n_var is not None:
            exo = self._create_exogene(idx)
            tmp_sample = np.concatenate((tmp_sample, exo), axis=0)

        if self.labels is None:
            return tmp_sample
        else:
            return tmp_sample.astype(np.float32), self.labels[idx]

    
    
class BasedSelfPretrainer(object):
    def __init__(self,
                 model,
                 train_loader, valid_loader=None,
                 learning_rate=1e-3, weight_decay=0,
                 name_scheduler=None,
                 dict_params_scheduler=None,
                 warmup_duration=None,
                 criterion=nn.MSELoss(), mask=None, loss_in_model=False,
                 device="cuda", all_gpu=False,
                 verbose=True, plotloss=True, 
                 save_fig=False, path_fig=None,
                 save_only_core=False,
                 save_checkpoint=False, path_checkpoint=None):

        # =======================class variables======================= #
        self.device = device
        self.all_gpu = all_gpu
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.mask = mask
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.save_only_core = save_only_core
        self.loss_in_model = loss_in_model
        self.name_scheduler = name_scheduler
        
        if name_scheduler is None:
            self.scheduler = None
        else:
            assert isinstance(dict_params_scheduler, dict)
            
            if name_scheduler=='MultiStepLR':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=dict_params_scheduler['milestones'], gamma=dict_params_scheduler['gamma'], verbose=self.verbose)

            elif name_scheduler=='CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=dict_params_scheduler['T_max'], eta_min=dict_params_scheduler['eta_min'], verbose=self.verbose)

            elif name_scheduler=='CosineAnnealingWarmRestarts':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=dict_params_scheduler['T_0'], T_mult=dict_params_scheduler['T_mult'], eta_min=dict_params_scheduler['eta_min'], verbose=self.verbose)

            elif name_scheduler=='ExponentialLR':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=dict_params_scheduler['gamma'], verbose=self.verbose)

            else:
                raise ValueError('Type of scheduler {} unknown, only "MultiStepLR", "ExponentialLR", "CosineAnnealingLR" or "CosineAnnealingWarmRestarts".'.format(name_scheduler))
        
        #if warmup_duration is not None:
        #    self.scheduler = create_lr_scheduler_with_warmup(scheduler,
        #                                                     warmup_start_value=1e-7,
        #                                                     warmup_end_value=learning_rate,
        #                                                     warmup_duration=warmup_duration)
        #else:
        #    self.scheduler = scheduler
            
        if self.all_gpu:
            # ===========dummy forward to intialize Lazy Module=========== #
            self.model.to("cpu")
            for ts in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # ===========data Parrallel Module call=========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model'

        self.log = {}
        self.train_time = 0
        self.passed_epochs = 0
        self.loss_train_history = []
        self.loss_valid_history = []
    
    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """
        t = time.time()
        for epoch in range(n_epochs):
            # =======================one epoch===================== #
            train_loss = self.__train(epoch)
            self.loss_train_history.append(train_loss)
            
            if self.valid_loader is not None:
                valid_loss = self.__evaluate()
                self.loss_valid_history.append(valid_loss)

            # =======================verbose======================= #
            if self.verbose:
                print('Epoch [{}/{}]'.format(epoch + 1, n_epochs))
                print('    Train loss : {:.6f}'.format(train_loss))
                if self.valid_loader is not None:
                    print('    Valid  loss : {:.6f}'.format(valid_loss))
            
            if epoch%5==0 or epoch==n_epochs-1:
                # =========================log========================= #
                if self.save_only_core:
                    self.log = {'model_state_dict': self.model.module.core.state_dict() if self.device=="cuda" and self.all_gpu else self.model.core.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss_train_history': self.loss_train_history,
                                'loss_valid_history': self.loss_valid_history,
                                'time': (time.time() - t)
                               }
                else:
                    self.log = {'model_state_dict': self.model.module.state_dict() if self.device=="cuda" and self.all_gpu else self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss_train_history': self.loss_train_history,
                                'loss_valid_history': self.loss_valid_history,
                                'time': (time.time() - t)
                               }
                    
                if self.save_checkpoint:
                    self.save()
                    
            if self.scheduler is not None: 
                if self.name_scheduler!='CosineAnnealingWarmRestarts':   
                    self.scheduler.step()
                
            self.passed_epochs+=1
            
        self.train_time = round((time.time() - t), 3)
        
        if self.save_checkpoint:
            self.save()

        if self.plotloss:
            self.plot_history()
            
        return
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return
    
    def plot_history(self):
        """
        Public function : plot loss history
        """
        fig = plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label='Train loss')
        if self.valid_loader is not None:
            plt.plot(range(self.passed_epochs), self.loss_valid_history, label='Valid loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        if self.save_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return
    
    def reduce_lr(self, new_lr):
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr
        return

    def __train(self, epoch):
        """
        Private function : model training loop over data loader
        """
        loss_train = 0
        iters = len(self.train_loader)
        
        for i, ts in enumerate(self.train_loader):
            self.model.train()
            # ===================variables=================== #
            ts = Variable(ts.float())
            if self.mask is not None:
                mask_loss, ts_masked = self.mask(ts)
            # ===================forward===================== #
            self.optimizer.zero_grad()
            if self.mask is not None:
                outputs = self.model(ts_masked.to(self.device))
                loss    = self.criterion(outputs, ts.to(self.device), mask_loss.to(self.device))
            else:
                if self.loss_in_model:
                    outputs, loss = self.model(ts.to(self.device))
                else:
                    outputs = self.model(ts.to(self.device))
                    loss    = self.criterion(outputs, ts.to(self.device))
            # ===================backward==================== #              
            loss.backward()
            self.optimizer.step()
            loss_train += loss.item()
            
            if self.name_scheduler=='CosineAnnealingWarmRestarts':
                self.scheduler.step(epoch + i / iters)

        loss_train = loss_train / len(self.train_loader)
        return loss_train
    
    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        loss_valid = 0
        with torch.no_grad():
            for ts in self.valid_loader:
                self.model.eval()
                # ===================variables=================== #
                ts = Variable(ts.float())
                if self.mask is not None:
                    mask_loss, ts_masked = self.mask(ts)
                # ===================forward===================== #
                if self.mask is not None:
                    outputs = self.model(ts_masked.to(self.device))
                    loss    = self.criterion(outputs, ts.to(self.device), mask_loss.to(self.device))
                else:
                    if self.loss_in_model:
                        outputs, loss = self.model(ts.to(self.device))
                    else:
                        outputs = self.model(ts.to(self.device))
                        loss    = self.criterion(outputs, ts.to(self.device))
                loss_valid += loss.item()

        loss_valid = loss_valid / len(self.valid_loader)
        return loss_valid


class BasedClassifTrainer_Sktime():
    def __init__(self,
                 model,
                 f_metrics=Classifmetrics(),
                 verbose=True, save_model=True,
                 save_checkpoint=False, path_checkpoint=None):
        """
        Trainer designed for scikit API like model and classification cases
        """
        self.model = model
        self.f_metrics = f_metrics
        self.verbose = verbose
        self.save_model = save_model
        self.save_checkpoint = save_checkpoint
        
        if path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model'
        
        self.train_time = 0
        self.test_time = 0
        self.log = {}
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None, instance_scaling=False):
        """
        Public function : fit API call
        
        -> Instance normalization by default
        """
        
        if instance_scaling:
            X_train = StandardScaler().fit_transform(X_train.T).T
        
        _t = time.time()
        self.model.fit(X_train, y_train.ravel())
        self.train_time = round((time.time() - _t), 3)
        self.log['time'] = self.train_time
        
        if self.save_model:
            self.log['model'] = self.model
            
        if X_valid is not None and y_valid is not None:
            if instance_scaling:
                X_valid = StandardScaler().fit_transform(X_valid.T).T
            valid_metrics = self.evaluate(X_valid, y_valid, mask='valid_metrics')

            if self.verbose:
                print('Valid metrics :', valid_metrics)
        
        if self.verbose:
            print('Training time :', self.train_time)
        
        return
    
    def evaluate(self, X_test, y_test, mask='test_metrics', instance_scaling=False):
        """
        Public function : predict API call then evaluation with given metric function
        
        -> Z-normalization of the data by default
        """
        
        if instance_scaling:
            X_test = StandardScaler().fit_transform(X_test.T).T
        
        _t = time.time()
        metrics = self.f_metrics(y_test.ravel(), self.model.predict(X_test))
        self.log[mask] = metrics
        self.test_time = round((time.time() - _t), 3)
        self.log[mask+'_time'] = self.test_time
        
        if self.save_checkpoint:
            self.save()

        return metrics
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return
    

class BasedClassifTrainer(object):
    def __init__(self,
                 model, 
                 train_loader, valid_loader=None,
                 learning_rate=1e-3, weight_decay=1e-2, 
                 criterion=nn.CrossEntropyLoss(),
                 patience_es=None, patience_rlr=None,
                 device="cuda", all_gpu=False,
                 valid_criterion=None,
                 n_warmup_epochs=0,
                 f_metrics=Classifmetrics(),
                 verbose=True, plotloss=True, 
                 save_fig=False, path_fig=None,
                 save_checkpoint=False, path_checkpoint=None):
        """
        PyTorch Model Trainer Class for classification case
        """

        # =======================class variables======================= #
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.f_metrics = f_metrics
        self.device = device
        self.all_gpu = all_gpu
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.patience_rlr = patience_rlr
        self.patience_es = patience_es
        self.n_warmup_epochs = n_warmup_epochs
        self.scheduler = None
        
        self.train_criterion = criterion
        if valid_criterion is None:
            self.valid_criterion = criterion
        else:
            self.valid_criterion = valid_criterion
        
        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model' 
            
        if patience_rlr is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                                        patience=patience_rlr, 
                                                                        verbose=self.verbose,
                                                                        eps=1e-7)
            
        #if n_warmup_epochs > 0 and self.scheduler is not None:
        #    self.scheduler = create_lr_scheduler_with_warmup(self.scheduler,
        #                                                     warmup_start_value=1e-6,
        #                                                     warmup_end_value=learning_rate,
        #                                                     warmup_duration=n_warmup_epochs)

        self.log = {}
        self.train_time = 0
        self.eval_time = 0
        self.voter_time = 0
        self.passed_epochs = 0
        self.best_loss = np.Inf
        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []
               
        if self.patience_es is not None:
            self.early_stopping = EarlyStopper(patience=self.patience_es)

        if self.all_gpu:
            # =========== Dummy forward to intialize Lazy Module if all GPU used =========== #
            self.model.to("cpu")
            for ts, _ in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # =========== Data Parrallel Module call =========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """
        
        #flag_es = 0
        tmp_time = time.time()
        
        for epoch in range(n_epochs):
            # =======================one epoch======================= #
            train_loss, train_accuracy = self.__train()
            self.loss_train_history.append(train_loss)
            self.accuracy_train_history.append(train_accuracy)
            if self.valid_loader is not None:
                valid_loss, valid_accuracy = self.__evaluate()
                self.loss_valid_history.append(valid_loss)
                self.accuracy_valid_history.append(valid_accuracy)
            else:
                valid_loss = train_loss
                
            # =======================reduce lr======================= #
            if self.scheduler:
                self.scheduler.step(valid_loss)

            # ===================early stoppping=================== #
            if self.patience_es is not None:
                if self.passed_epochs > self.n_warmup_epochs: # Avoid n_warmup_epochs first epochs
                    if self.early_stopping.early_stop(valid_loss):
                        #flag_es  = 1
                        es_epoch = epoch+1
                        self.passed_epochs+=1
                        if self.verbose:
                            print('Early stopping after {} epochs !'.format(epoch+1))
                        break
        
            # =======================verbose======================= #
            if self.verbose:
                print('Epoch [{}/{}]'.format(epoch+1, n_epochs))
                print('    Train loss : {:.4f}, Train acc : {:.2f}%'
                          .format(train_loss, train_accuracy*100))
                
                if self.valid_loader is not None:
                    print('    Valid  loss : {:.4f}, Valid  acc : {:.2f}%'
                              .format(valid_loss, valid_accuracy*100))

            # =======================save log======================= #
            if valid_loss <= self.best_loss and self.passed_epochs>=self.n_warmup_epochs:
                self.best_loss = valid_loss
                self.log = {'valid_metrics': valid_accuracy if self.valid_loader is not None else train_accuracy,
                            'model_state_dict': self.model.module.state_dict() if self.device=="cuda" and self.all_gpu else self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss_train_history': self.loss_train_history,
                            'loss_valid_history': self.loss_valid_history,
                            'accuracy_train_history': self.accuracy_train_history,
                            'accuracy_valid_history': self.accuracy_valid_history,
                            'value_best_loss': self.best_loss,
                            'epoch_best_loss': self.passed_epochs,
                            'time_best_loss': round((time.time() - tmp_time), 3),
                            }
                if self.save_checkpoint:
                    self.save()
                
            self.passed_epochs+=1
                    
        self.train_time = round((time.time() - tmp_time), 3)

        if self.plotloss:
            self.plot_history()
            
        if self.save_checkpoint:
            self.log['best_model_state_dict'] = torch.load(self.path_checkpoint+'.pt')['model_state_dict']
        
        # =======================update log======================= #
        self.log['training_time'] = self.train_time
        self.log['loss_train_history'] = self.loss_train_history
        self.log['loss_valid_history'] = self.loss_valid_history
        self.log['accuracy_train_history'] = self.accuracy_train_history
        self.log['accuracy_valid_history'] = self.accuracy_valid_history
        
        #if flag_es != 0:
        #    self.log['final_epoch'] = es_epoch
        #else:
        #    self.log['final_epoch'] = n_epochs
        
        if self.save_checkpoint:
            self.save()
        return
    
    def evaluate(self, test_loader, mask='test_metrics', return_output=False):
        """
        Public function : model evaluation on test dataset
        """
        tmp_time = time.time()
        mean_loss_eval = []
        y = np.array([])
        y_hat = np.array([])
        with torch.no_grad():
            for ts, labels in test_loader:
                self.model.eval()
                # ===================variables=================== #
                ts = Variable(ts.float()).to(self.device)
                labels = Variable(labels.float()).to(self.device)
                # ===================forward===================== #
                logits = self.model(ts)
                loss = self.valid_criterion(logits.float(), labels.long())
                # =================concatenate=================== #
                _, predicted = torch.max(logits, 1)
                mean_loss_eval.append(loss.item())
                y_hat = np.concatenate((y_hat, predicted.detach().cpu().numpy())) if y_hat.size else predicted.detach().cpu().numpy()
                y = np.concatenate((y, torch.flatten(labels).detach().cpu().numpy())) if y.size else torch.flatten(labels).detach().cpu().numpy()
                
        metrics = self.f_metrics(y, y_hat)
        self.eval_time = round((time.time() - tmp_time), 3)
        self.log[mask+'_time'] = self.eval_time
        self.log[mask] = metrics
        
        if self.save_checkpoint:
            self.save()
        
        if return_output:
            return np.mean(mean_loss_eval), metrics, y, y_hat
        else:
            return np.mean(mean_loss_eval), metrics
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return
    
    def plot_history(self):
        """
        Public function : plot loss history
        """
        fig = plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label='Train loss')
        if self.valid_loader is not None:
            plt.plot(range(self.passed_epochs), self.loss_valid_history, label='Valid loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        if self.path_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return
    
    def reduce_lr(self, new_lr):
        """
        Public function : update learning of the optimizer
        """
        for g in self.model.optimizer.param_groups:
            g['lr'] = new_lr
        return
            
    def restore_best_weights(self):
        """
        Public function : load best model state dict parameters met during training.
        """
        try:
            if self.all_gpu:
                self.model.module.load_state_dict(self.log['best_model_state_dict'])
            else:
                self.model.load_state_dict(self.log['best_model_state_dict'])
            print('Restored best model met during training.')
        except KeyError:
            print('Error during loading log checkpoint state dict : no update.')
        return
    
    def __train(self):
        """
        Private function : model training loop over data loader
        """
        total_sample_train = 0
        mean_loss_train = []
        mean_accuracy_train = []
        
        for ts, labels in self.train_loader:
            self.model.train()
            # ===================variables=================== #
            ts = Variable(ts.float()).to(self.device)
            labels = Variable(labels.float()).to(self.device)
            # ===================forward===================== #
            self.optimizer.zero_grad()
            logits = self.model(ts)
            # ===================backward==================== #
            loss_train = self.train_criterion(logits.float(), labels.long())
            loss_train.backward()
            self.optimizer.step()
            # ================eval on train================== #
            total_sample_train += labels.size(0)
            _, predicted_train = torch.max(logits, 1)
            correct_train = (predicted_train.to(self.device) == labels.to(self.device)).sum().item()
            mean_loss_train.append(loss_train.item())
            mean_accuracy_train.append(correct_train)
            
        return np.mean(mean_loss_train), np.sum(mean_accuracy_train)/total_sample_train
    
    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        total_sample_valid = 0
        mean_loss_valid = []
        mean_accuracy_valid = []
        
        with torch.no_grad():
            for ts, labels in self.valid_loader:
                self.model.eval()
                # ===================variables=================== #
                ts = Variable(ts.float()).to(self.device)
                labels = Variable(labels.float()).to(self.device)
                logits = self.model(ts)
                loss_valid = self.valid_criterion(logits.float(), labels.long())
                # ================eval on test=================== #
                total_sample_valid += labels.size(0)
                _, predicted = torch.max(logits, 1)
                correct = (predicted.to(self.device) == labels.to(self.device)).sum().item()
                mean_loss_valid.append(loss_valid.item())
                mean_accuracy_valid.append(correct)

        return np.mean(mean_loss_valid), np.sum(mean_accuracy_valid)/total_sample_valid 
    


class SeqToSeqTrainer():
    def __init__(self,
                 model, 
                 train_loader, valid_loader=None,
                 learning_rate=1e-3, weight_decay=1e-2,
                 criterion=nn.MSELoss(),
                 consumption_pred=False, timestamp_pred=True,
                 patience_es=None, patience_rlr=None,
                 device="cuda", all_gpu=False,
                 valid_criterion=None,
                 training_in_model=False, loss_in_model=False, moe_training=False,
                 f_metrics=NILMmetrics(),
                 n_warmup_epochs=0,
                 verbose=True, plotloss=True, 
                 save_fig=False, path_fig=None,
                 save_checkpoint=False, path_checkpoint=None):
        """
        PyTorch Model Trainer Class for SeqToSeq NILM (per timestamps estimation)

        Can be either: classification, values in [0,1] or energy power estimation for each timesteps
        """
        
        # =======================class variables======================= #
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.all_gpu = all_gpu
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.patience_rlr = patience_rlr
        self.patience_es = patience_es
        self.n_warmup_epochs = n_warmup_epochs
        self.consumption_pred = consumption_pred
        self.timestamp_pred = timestamp_pred
        self.f_metrics = f_metrics
        self.loss_in_model = loss_in_model
        self.training_in_model = training_in_model
        self.moe_training = moe_training

        if self.training_in_model:
            assert hasattr(self.model, 'train_one_epoch')
        
        self.train_criterion = criterion
        if valid_criterion is None:
            self.valid_criterion = criterion
        else:
            self.valid_criterion = valid_criterion
        
        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model'
            
        if self.patience_rlr is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                                        patience=self.patience_rlr, 
                                                                        verbose=self.verbose,
                                                                        eps=1e-7)
  
        self.log = {}
        self.train_time = 0
        self.eval_time = 0
        self.voter_time = 0
        self.passed_epochs = 0
        self.best_loss = np.Inf
        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []
               
        if self.patience_es is not None:
            self.early_stopping = EarlyStopper(patience=self.patience_es)

        if self.all_gpu:
            # =========== Dummy forward to intialize Lazy Module =========== #
            self.model.to("cpu")
            for ts, _, _ in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # =========== Data Parrallel Module call =========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """
        
        #flag_es = 0
        tmp_time = time.time()
        
        for epoch in range(n_epochs):
            # =======================one epoch======================= #
            if self.training_in_model:
                self.model.train()
                if self.all_gpu:
                    train_loss = self.model.module.train_one_epoch(loader=self.train_loader, optimizer=self.optimizer, device=self.device)
                else:
                    train_loss = self.model.train_one_epoch(loader=self.train_loader, optimizer=self.optimizer, device=self.device)
            else:
                train_loss = self.__train()
            self.loss_train_history.append(train_loss)
            if self.valid_loader is not None:
                valid_loss = self.__evaluate()
                self.loss_valid_history.append(valid_loss)
            else:
                valid_loss = train_loss
                
            # =======================reduce lr======================= #
            if self.patience_rlr:
                self.scheduler.step(valid_loss)

            # ===================early stoppping=================== #
            if self.patience_es is not None:
                if self.passed_epochs > self.n_warmup_epochs: # Avoid n_warmup_epochs first epochs
                    if self.early_stopping.early_stop(valid_loss):
                        #flag_es  = 1
                        es_epoch = epoch+1
                        self.passed_epochs+=1
                        if self.verbose:
                            print('Early stopping after {} epochs !'.format(epoch+1))
                        break
        
            # =======================verbose======================= #
            if self.verbose:
                print('Epoch [{}/{}]'.format(epoch+1, n_epochs))
                print('    Train loss : {:.4f}'
                          .format(train_loss))
                
                if self.valid_loader is not None:
                    print('    Valid  loss : {:.4f}'
                              .format(valid_loss))

            # =======================save log======================= #
            if valid_loss <= self.best_loss and self.passed_epochs>=self.n_warmup_epochs:
                self.best_loss = valid_loss
                self.log = {'model_state_dict': self.model.module.state_dict() if self.device=="cuda" and self.all_gpu else self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss_train_history': self.loss_train_history,
                            'loss_valid_history': self.loss_valid_history,
                            'value_best_loss': self.best_loss,
                            'epoch_best_loss': self.passed_epochs,
                            'time_best_loss': round((time.time() - tmp_time), 3),
                            }
                if self.save_checkpoint:
                    self.save()
                
            self.passed_epochs+=1
                    
        self.train_time = round((time.time() - tmp_time), 3)

        if self.plotloss:
            self.plot_history()

        if self.save_checkpoint:
            self.log['best_model_state_dict'] = torch.load(self.path_checkpoint+'.pt')['model_state_dict']
        
        # =======================update log======================= #
        self.log['training_time'] = self.train_time
        self.log['loss_train_history'] = self.loss_train_history
        self.log['loss_valid_history'] = self.loss_valid_history
        
        if self.save_checkpoint:
            self.save()
        return
    
    def evaluate(self, test_loader, inverse_scaling=False, scaler=None, threshold_small_values=None, save_outputs=False, mask='test_metrics'):
        """
        Public function : model evaluation on test dataset
        """
        tmp_time = time.time()
        loss_valid = 0

        if inverse_scaling:
            assert scaler is not None
        
        y = np.array([])
        y_states = np.array([])
        y_hat = np.array([])
        
        with torch.no_grad():
            for ts_agg, appl, states in test_loader:
                self.model.eval()
                if self.loss_in_model:
                    target = torch.Tensor(appl).float().to(self.device) 
                # ===================variables=================== #
                ts_agg = Variable(ts_agg.float()).to(self.device)
                if self.consumption_pred:
                    if inverse_scaling:
                        appl = scaler.inverse_transform_appliance(appl)

                    # Energy estimation
                    true_val = Variable(appl.float()).to(self.device)
                        
                    if not self.timestamp_pred:
                        # Energy estimation for the entire window (sum of all energy on the subsequence)
                        true_val = true_val.sum(dim=-1)

                    y_states = np.concatenate((y_states, appl.flatten())) if y_states.size else appl.flatten()
                else:
                    # Classification
                    true_val = Variable(states.float()).to(self.device)

                    if not self.timestamp_pred:
                        # Applicance detection on entire window
                        true_val = torch.where(true_val > 0, 1.0, 0.0)
                
                # True labels concatenation
                y = np.concatenate((y, torch.flatten(true_val).detach().cpu().numpy())) if y.size else torch.flatten(true_val).detach().cpu().numpy()
                
                # ===================forward===================== #
                if self.loss_in_model:
                    pred, _ = self.model(ts_agg, target)
                else:
                    pred = self.model(ts_agg)
                
                if self.consumption_pred:
                    # Clamp with minimum value to 0
                    pred = pred.clamp(min=0)

                    if inverse_scaling:
                        pred = scaler.inverse_transform_appliance(pred)
                        if threshold_small_values is not None:
                            pred[pred<threshold_small_values] = 0 

                        if scaler.appliance_scaling_type=='MaxScaling':
                            for idx_appl in range(pred.shape[1]):
                                pred[:, idx_appl, :] = pred[:, idx_appl, :].clamp(min=0, max=scaler.appliance_stat2[idx_appl])

                    if pred.shape[-1]>1 and not self.timestamp_pred:
                        pred = pred.sum(dim=-1)

                else:
                    pred = nn.Sigmoid()(pred)

                # TODO : change for multiple appliance
                y_hat = np.concatenate((y_hat, torch.flatten(pred).detach().cpu().numpy())) if y_hat.size else torch.flatten(pred).detach().cpu().numpy()
                
                loss = self.valid_criterion(pred, true_val)
                loss_valid += loss.item()

        loss_valid = loss_valid / len(self.valid_loader)
        
        if self.consumption_pred and self.timestamp_pred:
            metrics = self.f_metrics(y, y_hat, y_states, threshold_activation=10 if threshold_small_values is None else threshold_small_values)
        else:
            metrics = self.f_metrics(y, y_hat)

        self.eval_time = round((time.time() - tmp_time), 3)
        self.log[mask+'_time'] = self.eval_time
        self.log[mask] = metrics

        if save_outputs:
            self.log[mask+'_y'] = y
            self.log[mask+'_y_hat'] = y_hat
        
        if self.save_checkpoint:
            self.save()
        
        return np.mean(loss_valid), metrics
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return
    
    def plot_history(self):
        """
        Public function : plot loss history
        """
        fig = plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label='Train loss')
        if self.valid_loader is not None:
            plt.plot(range(self.passed_epochs), self.loss_valid_history, label='Valid loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        if self.path_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return
    
    def reduce_lr(self, new_lr):
        """
        Public function : update learning of the optimizer
        """
        for g in self.model.optimizer.param_groups:
            g['lr'] = new_lr
            
        return
            
    def restore_best_weights(self):
        """
        Public function : load best model state dict parameters met during training.
        """
        try:
            if self.all_gpu:
                self.model.module.load_state_dict(self.log['best_model_state_dict'])
            else:
                self.model.load_state_dict(self.log['best_model_state_dict'])
            print('Restored best model met during training.')
        except KeyError:
            print('Error during loading log checkpoint state dict : no update.')
        return
    
    def __train(self):
        """
        Private function : model training loop over data loader
        """
        loss_train = 0
        
        for ts_agg, appl, states in self.train_loader:
            self.model.train()
            
            if self.loss_in_model and ts_agg.shape[1] > 1:
                true_val_loss =  np.concatenate((ts_agg[:, 1:, :], appl), axis=1)
                true_val_loss = torch.Tensor(true_val_loss).float().to(self.device)  

            # ===================variables=================== #
            ts_agg = Variable(ts_agg.float()).to(self.device)    
            if self.consumption_pred:
                true_val = Variable(appl.float()).to(self.device)
                if not self.timestamp_pred:
                    true_val = true_val.sum(dim=-1)
            else:
                true_val = Variable(states.float()).to(self.device)
            # ===================forward===================== #
            self.optimizer.zero_grad()
            if self.moe_training:
                    pred, moe_loss = self.model(ts_agg)
                    loss = self.train_criterion(pred, true_val)
                    loss = loss + moe_loss * 0.01
            else:
                if self.loss_in_model:
                    pred, loss = self.model(ts_agg, true_val_loss)
                else:
                    pred = self.model(ts_agg)
                    if not self.consumption_pred and not self.timestamp_pred:
                        pred = nn.Sigmoid(pred)
                # ===================backward==================== #
                loss = self.train_criterion(pred, true_val)
            loss_train += loss.item()
            loss.backward()
            self.optimizer.step()

        loss_train = loss_train / len(self.train_loader)
            
        return loss_train
    
    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        loss_valid = 0
        
        with torch.no_grad():
            for ts_agg, appl, states in self.valid_loader:
                self.model.eval()
                if self.loss_in_model and ts_agg.shape[1] > 1:
                    true_val_loss =  np.concatenate((ts_agg[:, 1:, :], appl), axis=1)
                    true_val_loss = torch.Tensor(true_val_loss).float().to(self.device) 
                # ===================variables=================== #
                ts_agg = Variable(ts_agg.float()).to(self.device)    
                if self.consumption_pred:
                    true_val = Variable(appl.float()).to(self.device)
                    if not self.timestamp_pred:
                        true_val = true_val.sum(dim=-1)
                else:
                    true_val = Variable(states.float()).to(self.device)
                    if not self.timestamp_pred:
                        true_val = torch.where(true_val > 0, 1.0, 0.0)
                # ===================forward=================== #
                if self.loss_in_model:
                    pred, loss = self.model(ts_agg, true_val_loss)
                else:
                    pred = self.model(ts_agg)
                    loss = self.valid_criterion(pred, true_val)
                
                loss_valid += loss.item()
                
        loss_valid = loss_valid / len(self.valid_loader)
                
        return loss_valid
    

class SeqToPointTrainer():
    def __init__(self,
                 model, 
                 train_loader, valid_loader=None,
                 learning_rate=1e-3, weight_decay=1e-2,
                 criterion=nn.MSELoss(),
                 consumption_pred=False, timestamp_pred=True,
                 patience_es=None, patience_rlr=None,
                 device="cuda", all_gpu=False,
                 valid_criterion=None,
                 training_in_model=False, loss_in_model=False, moe_training=False,
                 f_metrics=NILMmetrics(),
                 n_warmup_epochs=0,
                 verbose=True, plotloss=True, 
                 save_fig=False, path_fig=None,
                 save_checkpoint=False, path_checkpoint=None):
        """
        PyTorch Model Trainer Class for SeqToPoint NILM (Regression) 
        """
        
        # =======================class variables======================= #
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.all_gpu = all_gpu
        self.verbose = verbose
        self.plotloss = plotloss
        self.save_checkpoint = save_checkpoint
        self.path_checkpoint = path_checkpoint
        self.save_fig = save_fig
        self.path_fig = path_fig
        self.patience_rlr = patience_rlr
        self.patience_es = patience_es
        self.n_warmup_epochs = n_warmup_epochs
        self.consumption_pred = consumption_pred
        self.timestamp_pred = timestamp_pred
        self.f_metrics = f_metrics
        self.loss_in_model = loss_in_model
        self.training_in_model = training_in_model
        self.moe_training = moe_training

        if self.training_in_model:
            assert hasattr(self.model, 'train_one_epoch')
        
        self.train_criterion = criterion
        if valid_criterion is None:
            self.valid_criterion = criterion
        else:
            self.valid_criterion = valid_criterion
        
        if self.path_checkpoint is not None:
            self.path_checkpoint = path_checkpoint
        else:
            self.path_checkpoint = os.getcwd()+os.sep+'model'
            
        if self.patience_rlr is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 
                                                                        patience=self.patience_rlr, 
                                                                        verbose=self.verbose,
                                                                        eps=1e-7)
  
        self.log = {}
        self.train_time = 0
        self.eval_time = 0
        self.voter_time = 0
        self.passed_epochs = 0
        self.best_loss = np.Inf
        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []
               
        if self.patience_es is not None:
            self.early_stopping = EarlyStopper(patience=self.patience_es)

        if self.all_gpu:
            # =========== Dummy forward to intialize Lazy Module =========== #
            self.model.to("cpu")
            for ts, _, _ in train_loader:
                self.model(torch.rand(ts.shape))
                break
            # =========== Data Parrallel Module call =========== #
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
    
    def train(self, n_epochs=10):
        """
        Public function : master training loop over epochs
        """
        
        tmp_time = time.time()
        
        for epoch in range(n_epochs):
            # =======================one epoch======================= #
            if self.training_in_model:
                self.model.train()
                if self.all_gpu:
                    train_loss = self.model.module.train_one_epoch(loader=self.train_loader, optimizer=self.optimizer, device=self.device)
                else:
                    train_loss = self.model.train_one_epoch(loader=self.train_loader, optimizer=self.optimizer, device=self.device)
            else:
                train_loss = self.__train()
            self.loss_train_history.append(train_loss)
            if self.valid_loader is not None:
                valid_loss = self.__evaluate()
                self.loss_valid_history.append(valid_loss)
            else:
                valid_loss = train_loss
                
            # =======================reduce lr======================= #
            if self.patience_rlr:
                self.scheduler.step(valid_loss)

            # ===================early stoppping=================== #
            if self.patience_es is not None:
                if self.passed_epochs > self.n_warmup_epochs: # Avoid n_warmup_epochs first epochs
                    if self.early_stopping.early_stop(valid_loss):
                        es_epoch = epoch+1
                        self.passed_epochs+=1
                        if self.verbose:
                            print('Early stopping after {} epochs !'.format(epoch+1))
                        break
        
            # =======================verbose======================= #
            if self.verbose:
                print('Epoch [{}/{}]'.format(epoch+1, n_epochs))
                print('    Train loss : {:.4f}'
                          .format(train_loss))
                
                if self.valid_loader is not None:
                    print('    Valid  loss : {:.4f}'
                              .format(valid_loss))

            # =======================save log======================= #
            if valid_loss <= self.best_loss and self.passed_epochs>=self.n_warmup_epochs:
                self.best_loss = valid_loss
                self.log = {'model_state_dict': self.model.module.state_dict() if self.device=="cuda" and self.all_gpu else self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss_train_history': self.loss_train_history,
                            'loss_valid_history': self.loss_valid_history,
                            'value_best_loss': self.best_loss,
                            'epoch_best_loss': self.passed_epochs,
                            'time_best_loss': round((time.time() - tmp_time), 3),
                            }
                if self.save_checkpoint:
                    self.save()
                
            self.passed_epochs+=1
                    
        self.train_time = round((time.time() - tmp_time), 3)

        if self.plotloss:
            self.plot_history()

        if self.save_checkpoint:
            self.log['best_model_state_dict'] = torch.load(self.path_checkpoint+'.pt')['model_state_dict']
        
        # =======================update log======================= #
        self.log['training_time'] = self.train_time
        self.log['loss_train_history'] = self.loss_train_history
        self.log['loss_valid_history'] = self.loss_valid_history
        
        if self.save_checkpoint:
            self.save()
        return
    
    def evaluate(self, test_loader, inverse_scaling=False, scaler=None, threshold_small_values=None, save_outputs=False, mask='test_metrics'):
        """
        Public function : model evaluation on test dataset
        """
        tmp_time = time.time()
        loss_valid = 0

        if inverse_scaling:
            assert scaler is not None
        
        y = np.array([])
        y_states = np.array([])
        y_hat = np.array([])
        
        with torch.no_grad():
            for ts_agg, appl, states in test_loader:
                self.model.eval()
                if self.loss_in_model and ts_agg.shape[1] > 1:
                    true_val_loss =  np.concatenate((ts_agg[:, 1:, :], appl), axis=1)
                    true_val_loss = torch.Tensor(true_val_loss).float().to(self.device) 
                # ===================variables=================== #
                ts_agg = Variable(ts_agg.float()).to(self.device)
                if self.consumption_pred:
                    if inverse_scaling:
                        appl = scaler.inverse_transform_appliance(appl)

                    # Energy estimation
                    true_val = Variable(appl.float()).to(self.device)
                        
                    if not self.timestamp_pred:
                        # Energy estimation for the entire window (sum of all energy on the subsequence)
                        true_val = true_val.sum(dim=-1)

                    y_states = np.concatenate((y_states, appl.flatten())) if y_states.size else appl.flatten()
                else:
                    # Classification
                    true_val = Variable(states.float()).to(self.device)

                    if not self.timestamp_pred:
                        # Applicance detection on entire window
                        true_val = torch.where(true_val > 0, 1.0, 0.0)
                
                # True labels concatenation
                y = np.concatenate((y, torch.flatten(true_val).detach().cpu().numpy())) if y.size else torch.flatten(true_val).detach().cpu().numpy()
                
                # ===================forward===================== #
                if self.loss_in_model:
                    pred, _ = self.model(ts_agg, true_val_loss)
                else:
                    pred = self.model(ts_agg)
                
                if self.consumption_pred:
                    # Clamp with minimum value to 0
                    pred = pred.clamp(min=0)

                    if inverse_scaling:
                        pred = scaler.inverse_transform_appliance(pred)
                        if threshold_small_values is not None:
                            pred[pred<threshold_small_values] = 0 

                        if scaler.appliance_scaling_type=='MaxScaling':
                            for idx_appl in range(pred.shape[1]):
                                pred[:, idx_appl, :] = pred[:, idx_appl, :].clamp(min=0, max=scaler.appliance_stat2[idx_appl])

                    if pred.shape[-1]>1 and not self.timestamp_pred:
                        pred = pred.sum(dim=-1)

                else:
                    pred = nn.Sigmoid()(pred)

                # TODO : change for multiple appliance
                y_hat = np.concatenate((y_hat, torch.flatten(pred).detach().cpu().numpy())) if y_hat.size else torch.flatten(pred).detach().cpu().numpy()
                
                loss = self.valid_criterion(pred, true_val)
                loss_valid += loss.item()

        loss_valid = loss_valid / len(self.valid_loader)
        
        if self.consumption_pred and self.timestamp_pred:
            metrics = self.f_metrics(y, y_hat, y_states, threshold_activation=10 if threshold_small_values is None else threshold_small_values)
        else:
            metrics = self.f_metrics(y, y_hat)

        self.eval_time = round((time.time() - tmp_time), 3)
        self.log['_time'] = self.eval_time
        self.log[mask] = metrics

        if save_outputs:
            self.log[mask+'_y'] = y
            self.log[mask+'_y_hat'] = y_hat
        
        if self.save_checkpoint:
            self.save()
        
        return np.mean(loss_valid), metrics
    
    def save(self):
        """
        Public function : save log
        """
        torch.save(self.log, self.path_checkpoint+'.pt')
        return
    
    def plot_history(self):
        """
        Public function : plot loss history
        """
        fig = plt.figure()
        plt.plot(range(self.passed_epochs), self.loss_train_history, label='Train loss')
        if self.valid_loader is not None:
            plt.plot(range(self.passed_epochs), self.loss_valid_history, label='Valid loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        if self.path_fig:
            plt.savefig(self.path_fig)
        else:
            plt.show()
        return
    
    def reduce_lr(self, new_lr):
        """
        Public function : update learning of the optimizer
        """
        for g in self.model.optimizer.param_groups:
            g['lr'] = new_lr
            
        return
            
    def restore_best_weights(self):
        """
        Public function : load best model state dict parameters met during training.
        """
        try:
            if self.all_gpu:
                self.model.module.load_state_dict(self.log['best_model_state_dict'])
            else:
                self.model.load_state_dict(self.log['best_model_state_dict'])
            print('Restored best model met during training.')
        except KeyError:
            print('Error during loading log checkpoint state dict : no update.')
        return
    
    def __train(self):
        """
        Private function : model training loop over data loader
        """
        loss_train = 0
        
        for ts_agg, appl, states in self.train_loader:
            self.model.train()

            # ===================variables=================== #
            ts_agg = Variable(ts_agg.float()).to(self.device)  

            if self.consumption_pred:
                true_val = Variable(appl.float()).to(self.device)
                true_val = true_val.sum(dim=-1)
            else:
                true_val = (states > 0).astype(dtype=int)
                true_val = Variable(true_val.float()).to(self.device)

            # ===================forward===================== #
            self.optimizer.zero_grad()
            if self.moe_training:
                    pred, moe_loss = self.model(ts_agg)
                    loss = self.train_criterion(pred, true_val)
                    loss = loss + moe_loss * 0.01
            else:
                pred = self.model(ts_agg)

                if not self.consumption_pred:
                    pred = nn.Sigmoid(pred)

                loss = self.train_criterion(pred, true_val)

            # ===================backward==================== #
            loss_train += loss.item()
            loss.backward()
            self.optimizer.step()

        loss_train = loss_train / len(self.train_loader)
            
        return loss_train
    
    def __evaluate(self):
        """
        Private function : model evaluation loop over data loader
        """
        loss_valid = 0
        
        with torch.no_grad():
            for ts_agg, appl, states in self.valid_loader:
                self.model.eval()
                if self.loss_in_model and ts_agg.shape[1] > 1:
                    true_val_loss =  np.concatenate((ts_agg[:, 1:, :], appl), axis=1)
                    true_val_loss = torch.Tensor(true_val_loss).float().to(self.device) 
                # ===================variables=================== #
                ts_agg = Variable(ts_agg.float()).to(self.device)    
                if self.consumption_pred:
                    true_val = Variable(appl.float()).to(self.device)
                    if not self.timestamp_pred:
                        true_val = true_val.sum(dim=-1)
                else:
                    true_val = Variable(states.float()).to(self.device)
                    if not self.timestamp_pred:
                        true_val = torch.where(true_val > 0, 1.0, 0.0)
                # ===================forward=================== #
                if self.loss_in_model:
                    pred, loss = self.model(ts_agg, true_val_loss)
                else:
                    pred = self.model(ts_agg)
                    loss = self.valid_criterion(pred, true_val)
                
                loss_valid += loss.item()
                
        loss_valid = loss_valid / len(self.valid_loader)
                
        return loss_valid

        
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
