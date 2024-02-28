import os
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler


# ================== Data Processing ================== #

def RandomUnderSampler_(X, y=None, sampling_strategy='auto', seed=0, nb_label=1):
    np.random.seed(seed)
    
    if isinstance(X, pd.core.frame.DataFrame):
        col = X.columns
        y = X.values[:, -nb_label].astype(int)
        X = X.values[:, :-nb_label]
        X_, y_ = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)
        Mat = np.concatenate((X_, np.reshape(y_, (y_.shape[0],  1))), axis=1)
        Mat = pd.DataFrame(data=Mat, columns=col)
        Mat = Mat.sample(frac=1, random_state=seed)
        
        return Mat
    else:
        assert y is not None, f"For np.array, please provide an y vector."
        X_, y_ = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed).fit_resample(X, y)
        Mat = np.concatenate((X_, np.reshape(y_, (y_.shape[0],  1))), axis=1)
        np.random.shuffle(Mat)
        Mat = Mat.astype(np.float32)
        
        return Mat[:, :-1], Mat[:, -1]

    
def split_train_valid_test(data, test_size=0.2, valid_size=0, nb_label_col=1):
    
    if isinstance(data, pd.core.frame.DataFrame):
        if valid_size != 0:
            X_train_valid, X_test, y_train_valid, y_test = train_test_split(data.iloc[:,:-nb_label_col], data.iloc[:,-nb_label_col:], test_size=test_size, random_state=0)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=valid_size, random_state=0)
                  
            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-nb_label_col], data.iloc[:,-nb_label_col:], test_size=test_size, random_state=0)
            return X_train, y_train, X_test, y_test
                  
    elif isinstance(data, np.ndarray):
        if valid_size != 0:
            X_train_valid, X_test, y_train_valid, y_test = train_test_split(data[:,:-nb_label_col], data[:,-nb_label_col:], test_size=test_size, random_state=0)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=valid_size, random_state=0)
            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(data[:,:-nb_label_col], data[:,-nb_label_col:], test_size=test_size, random_state=0)

            return X_train, y_train, X_test, y_test
    else:
        raise Exception('Please provide pandas Dataframe or numpy array object.')
        
        
def split_train_valid_test_pdl(df_data, test_size=0.2, valid_size=0, nb_label_col=1, seed=0, return_df=False):
    """
    Split DataFrame based on index ID (ID PDL for example)
    
    - Input : df_data -> DataFrame
              test_size -> Percentage data for test
              valid_size -> Percentage data for valid
              nb_label_col -> Number of columns of label
              seed -> Set seed
              return_df -> Return DataFrame instances, or Numpy Instances
    - Output:
            np.arrays or DataFrame Instances
    """

    np.random.seed(seed)
    list_pdl = np.array(df_data.index.unique())
    np.random.shuffle(list_pdl)
    pdl_train_valid = list_pdl[:int(len(list_pdl) * (1-test_size))]
    pdl_test = list_pdl[int(len(list_pdl) * (1-test_size)):]
    np.random.shuffle(pdl_train_valid)
    pdl_train = pdl_train_valid[:int(len(pdl_train_valid) * (1-valid_size))]
    
    df_train = df_data.loc[pdl_train, :].copy()
    df_test = df_data.loc[pdl_test, :].copy()
    

    df_train = df_train.sample(frac=1, random_state=seed)
    df_test = df_test.sample(frac=1, random_state=seed)
    
    if valid_size != 0:
        pdl_valid = pdl_train_valid[int(len(pdl_train_valid) * (1-valid_size)):]
        df_valid = df_data.loc[pdl_valid, :].copy()
        df_valid = df_valid.sample(frac=1, random_state=seed)
            
    if return_df:
        if valid_size != 0:
            return df_train, df_valid, df_test
        else:
            return df_train, df_test
    else:
        X_train = df_train.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
        y_train = df_train.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)
        X_test  = df_test.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
        y_test  = df_test.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)

        if valid_size != 0:
            X_valid = df_valid.iloc[:,:-nb_label_col].to_numpy().astype(np.float32)
            y_valid = df_valid.iloc[:,-nb_label_col:].to_numpy().astype(np.float32)

            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            return X_train, y_train, X_test, y_test


def Split_train_test_pdl_NILMDataset(data, st_date, seed=0,
                                     nb_house_test=None,  perc_house_test=None,  
                                     nb_house_valid=None, perc_house_valid=None):
    
    assert nb_house_test is not None or perc_house_test is not None
    assert len(data)==len(st_date)
    assert isinstance(st_date, pd.DataFrame)
    
    np.random.seed(seed)
    
    if nb_house_valid is not None or perc_house_valid is not None:
        assert (nb_house_test is not None and nb_house_valid is not None) or (perc_house_test is not None and perc_house_valid is not None)
    
    if len(data.shape) > 2:
        tmp_shape = data.shape
        data = data.reshape(data.shape[0], -1)
        
    data = pd.concat([st_date.reset_index(), pd.DataFrame(data)], axis=1).set_index('index')
    list_pdl = np.array(data.index.unique())
    np.random.shuffle(list_pdl)
    
    if nb_house_test is None:
        nb_house_test = max(1, int(len(list_pdl) * perc_house_test))
        if perc_house_valid is not None and nb_house_valid is None:
            nb_house_valid = max(1, int(len(list_pdl) * perc_house_valid))
        
    if nb_house_valid is not None:
        assert len(list_pdl) > nb_house_test + nb_house_valid
    else:
        assert len(list_pdl) > nb_house_test
    
    pdl_test = list_pdl[:nb_house_test]
    
    if nb_house_valid is not None:
        pdl_valid = list_pdl[nb_house_test:nb_house_test+nb_house_valid]
        pdl_train = list_pdl[nb_house_test+nb_house_valid:]
    else:
        pdl_train  = list_pdl[nb_house_test:]
    
    df_train = data.loc[pdl_train, :].copy()
    df_test  = data.loc[pdl_test, :].copy()
    
    st_date_train = df_train.iloc[:, :1]
    data_train    = df_train.iloc[:, 1:].values.reshape((len(df_train), tmp_shape[1], tmp_shape[2], tmp_shape[3]))
    st_date_test  = df_test.iloc[:, :1]
    data_test     = df_test.iloc[:, 1:].values.reshape((len(df_test), tmp_shape[1], tmp_shape[2], tmp_shape[3]))
    
    if nb_house_valid is not None:
        df_valid      = data.loc[pdl_valid, :].copy()
        st_date_valid = df_valid.iloc[:, :1]
        data_valid    = df_valid.iloc[:, 1:].values.reshape((len(df_valid), tmp_shape[1], tmp_shape[2], tmp_shape[3]))
        
        return data_train, st_date_train, data_valid, st_date_valid, data_test, st_date_test
    else:
        return data_train, st_date_train, data_test, st_date_test
        

# ===================== UKDALE DataBuilder =====================#
class UKDALE_DataBuilder(object):
    def __init__(self, 
                 data_path,
                 mask_app,
                 sampling_rate,
                 window_size,
                 window_stride=None,
                 soft_label=False):
        
        # =============== Class variables =============== #
        self.data_path = data_path
        self.mask_app = mask_app
        self.sampling_rate = sampling_rate 
        self.window_size = window_size
        self.soft_label = soft_label
        
        if isinstance(self.mask_app, str):
            self.mask_app = [self.mask_app]
        
        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        # ======= Add aggregate to appliance(s) list ======= #
        self._check_appliance_names()
        self.mask_app = ['aggregate'] + self.mask_app

        # ======= Dataset Parameters ======= #
        self.cutoff = 6000

        # All parameters are in Watts and Second
        self.appliance_param = {'kettle': {'min_threshold': 500, 'max_threshold': 5000, 'min_on_duration': 10, 'min_off_duration': 1},
                                'washing_machine': {'min_threshold': 20, 'max_threshold': 5000, 'min_on_duration': 300, 'min_off_duration': 1},
                                'dishwasher': {'min_threshold': 10, 'max_threshold': 5000, 'min_on_duration': 300, 'min_off_duration': 1},
                                'microwave': {'min_threshold': 200, 'max_threshold': 3000, 'min_on_duration': 10, 'min_off_duration': 30},
                                'fridge': {'min_threshold': 50, 'max_threshold': 300, 'min_on_duration': 60, 'min_off_duration': 12},
                                'cooker': {'min_threshold': 50, 'max_threshold': 6000, 'min_on_duration': 60, 'min_off_duration': 1},
                                'electric_heater': {'min_threshold': 50, 'max_threshold': 6000, 'min_on_duration': 60, 'min_off_duration': 1}
                                }

    def get_house_data(self, house_indicies):

        assert len(house_indicies)==1, f'get_house_data() implemented to get data from 1 house only at a time.'

        return self._get_dataframe(house_indicies[0])

    def get_classif_dataset(self, house_indicies):
        """
        Process data to build classif dataset
        
        Return : 
            -Time Series: np.ndarray of size [N_ts, Win_Size]
            -Associated label for each subsequences: np.ndarray of size [N_ts, 1] in 0 or 1 for each TS
            -st_date: pandas.Dataframe as :
                - index: id of each house
                - column 'start_date': Starting timestamp of each TS
        """

        nilm_dataset, st_date = self.get_nilm_dataset(house_indicies)
        y = np.zeros(len(nilm_dataset))

        for idx in range(len(nilm_dataset)):    
            if (nilm_dataset[idx, 1, 1, :] > 0).any():
                y[idx] = 1

        return nilm_dataset[:, 0, 0, :], y, st_date        
        
    def get_nilm_dataset(self, house_indicies):
        """
        Process data to build NILM usecase
        
        Return : 
            - np.ndarray of size [N_ts, M_appliances, 2, Win_Size] as :
        
                -1st dimension : nb ts obtained after slicing the total load curve of chosen Houses
                -2nd dimension : nb chosen appliances
                                -> indice 0 for aggregate load curve
                                -> Other appliance in same order as given "appliance_names" list
                -3rd dimension : access to load curve (values of consumption in Wh) or states of activation 
                                of the appliance (0 or 1 for each time step)
                                -> indice 0 : access to load curve
                                -> indice 1 : access to states of activation (0 or 1 for each time step) or Probability (i.e. value in [0, 1]) if soft_label
                -4th dimension : values

            - pandas.Dataframe as :
                index: id of each house
                column 'start_date': Starting timestamp of each TS
        """
        
        output_data = np.array([])
        st_date = pd.DataFrame()
        
        for indice in house_indicies:
            tmp_list_st_date = []
            
            data = self._get_dataframe(indice)
            stems, st_date_stems = self._get_stems(data)
            
            if self.window_size==self.window_stride:
                n_wins = len(data) // self.window_stride
            else:
                n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)
            
            X = np.empty((len(house_indicies) * n_wins, len(self.mask_app), 2, self.window_size))
            
            cpt = 0
            for i in range(n_wins):
                tmp = stems[:, i*self.window_stride:i*self.window_stride+self.window_size]

                if not self._check_anynan(tmp): # Check if nan -> skip the subsequences if it's the case
                    tmp_list_st_date.append(st_date_stems[i*self.window_stride])

                    X[cpt, 0, 0, :] = tmp[0, :]
                    X[cpt, 0, 1, :] = (tmp[0, :] > 0).astype(dtype=int)

                    key = 1
                    for j in range(1, len(self.mask_app)):
                        X[cpt, j, 0, :] = tmp[key, :]
                        X[cpt, j, 1, :] = tmp[key+1, :]
                        key += 2

                    cpt += 1 # Add one subsequence
                    
            tmp_st_date = pd.DataFrame(data=tmp_list_st_date, index=[indice for j in range(cpt)], columns=['start_date'])
            output_data = np.concatenate((output_data, X[:cpt, :, :, :]), axis=0) if output_data.size else X[:cpt, :, :, :]
            st_date = pd.concat([st_date, tmp_st_date], axis=0) if st_date.size else tmp_st_date
                        
        return output_data, st_date
    
    def _get_stems(self, dataframe):
        """
        Extract load curve for each chosen appliances.
        
        Return : np.ndarray instance
        """
        stems = np.empty((1 + (len(self.mask_app)-1)*2, dataframe.shape[0]))
        stems[0, :] = dataframe['aggregate'].values

        key = 1
        for appliance in self.mask_app[1:]:
            stems[key, :] = dataframe[appliance].values
            stems[key+1, :] = dataframe[appliance+'_status'].values
            key+=2

        return stems, list(dataframe.index)
    
    def _get_dataframe(self, indice):
        """
        Load houses data and return one dataframe with aggregate and appliance resampled at chosen time step.
        
        Return : pd.core.frame.DataFrame instance
        """
        path_house = self.data_path+'House'+str(indice)+os.sep
        self._check_if_file_exist(path_house+'labels.dat') # Check if labels exist at provided path
        
        # House labels
        house_label = pd.read_csv(path_house+'labels.dat',    sep=' ', header=None)
        house_label.columns = ['id', 'appliance_name']
        
        # Load aggregate load curve and resample to lowest sampling rate
        house_data = pd.read_csv(path_house+'channel_1.dat', sep=' ', header=None)
        house_data.columns = ['time','aggregate']
        house_data['time'] = pd.to_datetime(house_data['time'], unit = 's')
        house_data = house_data.set_index('time') # Set index to time
        house_data = house_data.resample('30s').mean().fillna(method='ffill', limit=30) # Resample to 6s for this dataset
        house_data[house_data < 5] = 0 # Remove small value

        """
        if indice==4:
            appl_data = pd.read_csv(path_house+'channel_6.dat', sep=' ', header=None)
            appl_data.columns = ['time', 'tb_remove']
            appl_data['time'] = pd.to_datetime(appl_data['time'],unit = 's')
            appl_data = appl_data.set_index('time')
            appl_data = appl_data.resample('30s').mean()
            appl_data[appl_data < 5] = 0 # Remove small value
            house_data = pd.merge(house_data, appl_data, how='inner', on='time')
            del appl_data
            house_data['aggregate'] = house_data['aggregate'] - house_data['tb_remove']
            house_data = house_data.drop('tb_remove', axis=1)
        """
        
        for appliance in self.mask_app[1:]:
            # Check if appliance is in this house
            if len(house_label.loc[house_label['appliance_name']==appliance]['id'].values) != 0:
                i = house_label.loc[house_label['appliance_name']==appliance]['id'].values[0]

                 # Load aggregate load curve and resample to lowest sampling rate
                appl_data = pd.read_csv(path_house+'channel_'+str(i)+'.dat', sep=' ', header=None)
                appl_data.columns = ['time', appliance]
                appl_data['time'] = pd.to_datetime(appl_data['time'],unit = 's')
                appl_data = appl_data.set_index('time')
                appl_data = appl_data.resample('30s').mean().fillna(method='ffill', limit=30) 
                appl_data[appl_data < 5] = 0 # Remove small value
                
                # Merge aggregate load curve with appliance load curve
                house_data = pd.merge(house_data, appl_data, how='inner', on='time')
                del appl_data
                house_data = house_data.clip(lower=0, upper=self.cutoff) # Apply general cutoff
                house_data = house_data.sort_index()

                # Replace nan values by -1 during appliance activation status filtering
                house_data[appliance].fillna(-1, inplace=True)

                # Create status column for this appliance 
                house_data[appliance+'_status'] = ((house_data[appliance] >= self.appliance_param[appliance]['min_threshold']) & (house_data[appliance] <= self.appliance_param[appliance]['max_threshold'])).astype(int)                

                # Filter activation with sufficient "ON" time
                activation_periods = []
                current_period_start = None
                for i, row in house_data.iterrows():
                    if row[appliance+'_status'] == 1:  # Appliance is on
                        if current_period_start is None:
                            current_period_start = i
                    else:  # Appliance is off
                        if current_period_start is not None:
                            duration = (i - current_period_start).seconds
                            if duration >= self.appliance_param[appliance]['min_on_duration']:
                                activation_periods.append((current_period_start, i))
                            current_period_start = None

                # Filter out activations with insufficient "OFF" time (during two activation)
                filtered_activation_periods = []
                for start, end in activation_periods:
                    if filtered_activation_periods:
                        last_end = filtered_activation_periods[-1][1]
                        if (start - last_end).seconds >= self.appliance_param[appliance]['min_off_duration']:
                            filtered_activation_periods.append((start, end))
                    else:
                        filtered_activation_periods.append((start, end))

                # Update status column based on activation periods
                house_data[appliance+'_status'] = 0
                for start, end in filtered_activation_periods:
                    house_data.loc[start:end, appliance+'_status'] = 1

                # Finally replacing nan values put to -1 by nan
                house_data[appliance].replace(-1, np.nan, inplace=True)

        if self.sampling_rate!='30s':
            house_data = house_data.resample(self.sampling_rate).mean()

        for appliance in self.mask_app[1:]:
            if appliance in house_data:
                if not self.soft_label:
                    house_data[appliance+'_status'] = (house_data[appliance+'_status'] > 0).astype(int)
                else:
                    continue
            else:
                house_data[appliance] = 0
                house_data[appliance+'_status'] = 0

        return house_data
    
    def _check_appliance_names(self):
        """
        Check appliances names for UKDALE case.
        """
        for appliance in self.mask_app:
            assert appliance in ['washing_machine', 'cooker', 'dishwasher', 'kettle', 'fridge', 'microwave', 'electric_heater'], f"Selected applicance unknow for UKDALE Dataset, got: {appliance}"
        return
            
    def _check_if_file_exist(self, file):
        """
        Check if file exist at provided path.
        """
        if os.path.isfile(file):
            pass
        else:
            raise FileNotFoundError
        return
    
    def _check_anynan(self, a):
        """
        Fast check of NaN in a numpy array.
        """
        return np.isnan(np.sum(a))
    

# ===================== REFIT DataBuilder =====================#
class REFIT_DataBuilder(object):
    def __init__(self, 
                 data_path,
                 mask_app,
                 sampling_rate,
                 window_size,
                 window_stride=None,
                 soft_label=False):
        
        # =============== Class variables =============== #
        self.data_path = data_path
        self.mask_app = mask_app
        self.sampling_rate = sampling_rate 
        self.window_size = window_size
        self.soft_label = soft_label
        
        if isinstance(self.mask_app, str):
            self.mask_app = [self.mask_app]
        
        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        # ======= Add aggregate to appliance(s) list ======= #
        self._check_appliance_names()
        self.mask_app = ['Aggregate'] + self.mask_app

        # ======= Dataset Parameters ======= #
        self.cutoff = 6000

        # All parameters are in Watts and Second
        self.appliance_param = {'Kettle': {'min_threshold': 500, 'max_threshold': 6000, 'min_on_duration': 30, 'min_off_duration': 1},
                                'WashingMachine': {'min_threshold': 300, 'max_threshold': 4000, 'min_on_duration': 300, 'min_off_duration': 1},
                                'Dishwasher': {'min_threshold': 300, 'max_threshold': 4000, 'min_on_duration': 300, 'min_off_duration': 1},
                                'Microwave': {'min_threshold': 200, 'max_threshold': 3000, 'min_on_duration': 30, 'min_off_duration': 30}
                                }

    def get_house_data(self, house_indicies):

        assert len(house_indicies)==1, f'get_house_data() implemented to get data from 1 house only at a time.'

        return self._get_dataframe(house_indicies[0])

    def get_classif_dataset(self, house_indicies):
        """
        Process data to build classif dataset
        
        Return : 
            -Time Series: np.ndarray of size [N_ts, Win_Size]
            -Associated label for each subsequences: np.ndarray of size [N_ts, 1] in 0 or 1 for each TS
            -st_date: pandas.Dataframe as :
                - index: id of each house
                - column 'start_date': Starting timestamp of each TS
        """

        nilm_dataset, st_date = self.get_nilm_dataset(house_indicies)
        y = np.zeros(len(nilm_dataset))

        for idx in range(len(nilm_dataset)):    
            if (nilm_dataset[idx, 1, 1, :] > 0).any():
                y[idx] = 1

        return nilm_dataset[:, 0, 0, :], y, st_date        
        
    def get_nilm_dataset(self, house_indicies):
        """
        Process data to build NILM usecase
        
        Return : 
            - np.ndarray of size [N_ts, M_appliances, 2, Win_Size] as :
        
                -1st dimension : nb ts obtained after slicing the total load curve of chosen Houses
                -2nd dimension : nb chosen appliances
                                -> indice 0 for aggregate load curve
                                -> Other appliance in same order as given "appliance_names" list
                -3rd dimension : access to load curve (values of consumption in Wh) or states of activation 
                                of the appliance (0 or 1 for each time step)
                                -> indice 0 : access to load curve
                                -> indice 1 : access to states of activation (0 or 1 for each time step) or Probability (i.e. value in [0, 1]) if soft_label
                -4th dimension : values

            - pandas.Dataframe as :
                index: id of each house
                column 'start_date': Starting timestamp of each TS
        """
        
        output_data = np.array([])
        st_date = pd.DataFrame()
        
        for indice in house_indicies:
            tmp_list_st_date = []
            
            data = self._get_dataframe(indice)
            stems, st_date_stems = self._get_stems(data)
            
            if self.window_size==self.window_stride:
                n_wins = len(data) // self.window_stride
            else:
                n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)
            
            X = np.empty((len(house_indicies) * n_wins, len(self.mask_app), 2, self.window_size))
            
            cpt = 0
            for i in range(n_wins):
                tmp = stems[:, i*self.window_stride:i*self.window_stride+self.window_size]

                if not self._check_anynan(tmp): # Check if nan -> skip the subsequences if it's the case
                    tmp_list_st_date.append(st_date_stems[i*self.window_stride])

                    X[cpt, 0, 0, :] = tmp[0, :]
                    X[cpt, 0, 1, :] = (tmp[0, :] > 0).astype(dtype=int)

                    key = 1
                    for j in range(1, len(self.mask_app)):
                        X[cpt, j, 0, :] = tmp[key, :]
                        X[cpt, j, 1, :] = tmp[key+1, :]
                        key += 2

                    cpt += 1 # Add one subsequence
                    
            tmp_st_date = pd.DataFrame(data=tmp_list_st_date, index=[indice for j in range(cpt)], columns=['start_date'])
            output_data = np.concatenate((output_data, X[:cpt, :, :, :]), axis=0) if output_data.size else X[:cpt, :, :, :]
            st_date = pd.concat([st_date, tmp_st_date], axis=0) if st_date.size else tmp_st_date
                        
        return output_data, st_date
    
    def _get_stems(self, dataframe):
        """
        Extract load curve for each chosen appliances.
        
        Return : np.ndarray instance
        """
        stems = np.empty((1 + (len(self.mask_app)-1)*2, dataframe.shape[0]))
        stems[0, :] = dataframe['Aggregate'].values

        key = 1
        for appliance in self.mask_app[1:]:
            stems[key, :] = dataframe[appliance].values
            stems[key+1, :] = dataframe[appliance+'_status'].values
            key+=2

        return stems, list(dataframe.index)
    
    def _get_dataframe(self, indice):
        """
        Load houses data and return one dataframe with aggregate and appliance resampled at chosen time step.
        
        Return : pd.core.frame.DataFrame instance
        """
        file = self.data_path+'CLEAN_House'+str(indice)+'.csv'
        self._check_if_file_exist(file)
        labels_houses = pd.read_csv(self.data_path+'HOUSES_Labels').set_index('House_id')

        house_data = pd.read_csv(file)
        house_data.columns = list(labels_houses.loc[int(indice)].values)
        house_data = house_data.set_index('Time').sort_index()
        house_data.index = pd.to_datetime(house_data.index)
        idx_to_drop = house_data[house_data['Issues']==1].index
        house_data = house_data.drop(index=idx_to_drop, axis=0)
        house_data = house_data.resample(rule='30s').mean().ffill(limit=30)
        house_data[house_data < 5] = 0 # Remove small value
        house_data = house_data.clip(lower=0, upper=self.cutoff) # Apply general cutoff
        house_data = house_data.sort_index()
        
        for appliance in self.mask_app[1:]:
            # Check if appliance is in this house
            if appliance in house_data:

                # Replace nan values by -1 during appliance activation status filtering
                house_data[appliance].fillna(-1, inplace=True)

                # Create status column for this appliance 
                house_data[appliance+'_status'] = ((house_data[appliance] >= self.appliance_param[appliance]['min_threshold']) & (house_data[appliance] <= self.appliance_param[appliance]['max_threshold'])).astype(int)                

                # Filter activation with sufficient "ON" time
                activation_periods = []
                current_period_start = None
                for i, row in house_data.iterrows():
                    if row[appliance+'_status'] == 1:  # Appliance is on
                        if current_period_start is None:
                            current_period_start = i
                    else:  # Appliance is off
                        if current_period_start is not None:
                            duration = (i - current_period_start).seconds
                            if duration >= self.appliance_param[appliance]['min_on_duration']:
                                activation_periods.append((current_period_start, i))
                            current_period_start = None

                # Filter out activations with insufficient "OFF" time (during two activation)
                filtered_activation_periods = []
                for start, end in activation_periods:
                    if filtered_activation_periods:
                        last_end = filtered_activation_periods[-1][1]
                        if (start - last_end).seconds >= self.appliance_param[appliance]['min_off_duration']:
                            filtered_activation_periods.append((start, end))
                    else:
                        filtered_activation_periods.append((start, end))

                # Update status column based on activation periods
                house_data[appliance+'_status'] = 0
                for start, end in filtered_activation_periods:
                    house_data.loc[start:end, appliance+'_status'] = 1

                # Finally replacing nan values put to -1 by nan
                house_data[appliance].replace(-1, np.nan, inplace=True)

        if self.sampling_rate!='30s':
            house_data = house_data.resample(self.sampling_rate).mean()

        tmp_list = ['Aggregate']
        for appliance in self.mask_app[1:]:
            tmp_list.append(appliance)
            tmp_list.append(appliance+'_status')
            if appliance in house_data:
                if not self.soft_label:
                    house_data[appliance+'_status'] = (house_data[appliance+'_status'] > 0).astype(int)
                else:
                    continue
            else:
                house_data[appliance] = 0
                house_data[appliance+'_status'] = 0

        house_data = house_data[tmp_list]

        return house_data
    
    def _check_appliance_names(self):
        """
        Check appliances names for UKDALE case.
        """
        for appliance in self.mask_app:
            assert appliance in ['WashingMachine', 'Dishwasher', 'Kettle', 'Microwave'], f"Selected applicance unknow for REFIT Dataset, got: {appliance}"
        return
            
    def _check_if_file_exist(self, file):
        """
        Check if file exist at provided path.
        """
        if os.path.isfile(file):
            pass
        else:
            raise FileNotFoundError
        return
    
    def _check_anynan(self, a):
        """
        Fast check of NaN in a numpy array.
        """
        return np.isnan(np.sum(a))
    

