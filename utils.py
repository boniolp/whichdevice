# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import textwrap
import streamlit as st

# === Lib import === #
import os, lzma, io
import numpy as np
import pandas as pd
import torch

# === Vizu import === #
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === Customs import === #
from constants import *
from Models.FCN import FCN
from Models.ResNet import ResNet
from Models.InceptionTime import Inception
from Helpers.class_activation_map import CAM

CURRENT_WINDOW=0

def run_playground_frame():

    global CURRENT_WINDOW
    
    st.markdown("Here show the time series and CAM")

    col1, col2 = st.columns(2)

    with col1:
        ts_name = st.selectbox(
            "Choose a load curve", list_name_ts, index=0
        )
    with col2:
        appliances = st.multiselect(
            "Choose devices:", devices_list, ["Dishwasher"]
        )

    col3_1, col3_2, col3_3 = st.columns(3)
    with col3_1:
        frequency = st.selectbox(
            "Choose a sampling rate:", frequency_list, index=0
        )
    with col3_2:
        models = st.multiselect(
            "Choose models:", models_list, ["ResNet"]
        )
    with col3_3:
        length = st.selectbox(
            "Choose the window length:", lengths_list, index=0
        )
    
    #st.markdown("show TS et prob devices")
    df, window_size = get_time_series_data(ts_name, frequency=frequency, length=length)
    n_win = len(df) // window_size

    colcontrol_1, colcontrol_2, colcontrol_3 = st.columns(3)
    with colcontrol_1:
        if st.button("Previous", type="primary"):
            CURRENT_WINDOW -= 1
            CURRENT_WINDOW  = max(0,CURRENT_WINDOW)
    with colcontrol_2:
        st.markdown("Window {}".format(CURRENT_WINDOW))
    
    with colcontrol_3:
        if st.button("Next", type="primary"):
            CURRENT_WINDOW += 1
            CURRENT_WINDOW  = min(CURRENT_WINDOW,n_win)
    
    pred_dict_all = pred_one_window(CURRENT_WINDOW, df, window_size, ts_name, appliances, frequency, models)
    fig_ts, fig_app, fig_prob = plot_one_window(CURRENT_WINDOW,  df, window_size, appliances, pred_dict_all)
    
    st.plotly_chart(fig_ts, use_container_width=True)
    st.plotly_chart(fig_app, use_container_width=True)
    st.plotly_chart(fig_prob, use_container_width=True)


    if st.button("When the appliance is used?", type="primary"):
        st.markdown("show CAM")
        fig_cam = plot_cam(CURRENT_WINDOW, df, window_size, appliances, pred_dict_all)
        st.plotly_chart(fig_cam, use_container_width=True)
        
            
    

def run_benchmark_frame():
    st.markdown("Here show benchmark results")


def run_about_frame():
    st.markdown("Here show info on the models, data and us")







def get_model_instance(model_name):
    # Load instance according to selected model
    if model_name=='ConvNet':
        model_inst = FCN()
    elif model_name=='ResNet':
        model_inst = ResNet()
    elif model_name=='Inception':
        model_inst = Inception()
    else:
        raise ValueError('Wrong model name.')

    return model_inst


def get_dataset_name(ts_name):
    # Get dataset_name according to choosen ts_name
    if 'UKDALE' in ts_name:
        dataset_name = 'UKDALE'
    elif 'REFIT' in ts_name:
        dataset_name = 'REFIT'
    else:
        raise ValueError('Wrong dataset name.')
    
    return dataset_name


def convert_length_to_window_size(frequency, length):
    # Dictionary to convert lengths to total minutes
    length_to_minutes = {
        '6 hours': 6 * 60,
        '12 hours': 12 * 60,
        '1 Day': 24 * 60
    }
    
    # Dictionary to convert frequency shorthand to total seconds
    freq_to_seconds = {
        '30s': 30,
        '1T': 60,
        '10T': 10 * 60
    }
    
    # Convert length to minutes
    if length in length_to_minutes:
        total_length_minutes = length_to_minutes[length]
    else:
        raise ValueError("Length not recognized. Please use '6 hours', '12 hours', or '1 Day'.")
    
    # Convert frequency to seconds
    if frequency in freq_to_seconds:
        frequency_seconds = freq_to_seconds[frequency]
    else:
        raise ValueError("Frequency not recognized. Please use '30 seconds', '1 minutes', or '10 minutes'.")
    
    # Calculate window size (total_length in seconds divided by frequency in seconds)
    # Ensure to convert minutes to seconds for total length
    window_size = (total_length_minutes * 60) / frequency_seconds
    
    return int(window_size)
    

def get_time_series_data(ts_name, frequency, length):
    dict_freq   = {'30 seconds': '30s', '1 minutes': '1T', '10 minutes': '10T'}
    pd_freq     = dict_freq[frequency]

    # Convert selected length to window_size according to choseen frequency
    window_size = convert_length_to_window_size(pd_freq, length)

    # Load dataframe
    df = pd.read_csv(os.getcwd()+f'/data/{ts_name}.gzip', compression='gzip', parse_dates=['Time']).set_index('Time')
    
    # Resample to choosen frequency (if > 30s)
    if pd_freq!='30s':
        df = df.resample(pd_freq).mean()

    return df, window_size


def get_prediction_one_appliance(ts_name, window_agg, appliance, frequency, model_list):
    dict_freq  = {'30 seconds': '30s', '1 minutes': '1T', '10 minutes': '10T'}
    sampling_rate = dict_freq[frequency]

    window_agg  = torch.Tensor(window_agg).unsqueeze(0).unsqueeze(0)

    pred_dict = {}
        
    for model_name in model_list:
        # Get model instance
        model_inst = get_model_instance(model_name)
        # Load compressed model
        path_model = os.getcwd()+f'/trained_clf/{get_dataset_name(ts_name)}/{sampling_rate}/{appliance}/{model_name}.pt.xz'
        # Decompress model
        with lzma.open(path_model, 'rb') as file:
            decompressed_file = file.read()
        model_parameters = torch.load(io.BytesIO(decompressed_file), map_location='cpu')
        del decompressed_file
        # Load state dict
        model_inst.load_state_dict(model_parameters['model_state_dict'])
        del model_parameters
        # Set model to eval mode
        model_inst.eval()

        # Predict proba and label
        pred_prob  = torch.nn.Softmax(dim=-1)(model_inst(window_agg)).detach().numpy().flatten()
        pred_label = np.argmax(pred_prob)

        # Predict CAM if Conv based architecture
        if model_name in ['ConvNet', 'ResNet', 'Inception']:
            pred_cam = get_cam(window_agg, model_name, model_inst)
        else:
            pred_cam = None

        # Update pred_dict
        pred_dict[model_name] = {'pred_prob': pred_prob, 'pred_label': pred_label, 'pred_cam': pred_cam}

    return pred_dict


def get_cam(window_agg, model_name, model_inst):

    # Set layer conv and fc layer names for selected model
    if model_name=='ConvNet':
        last_conv_layer = model_inst._modules['layer3']
        fc_layer_name   = model_inst._modules['linear']
    elif model_name=='ResNet':
        last_conv_layer = model_inst._modules['layers'][2]
        fc_layer_name   = model_inst._modules['linear']
    elif model_name=='Inception':
        last_conv_layer = model_inst._modules['Blocks'][1]
        fc_layer_name   = model_inst._modules['Linear']

    # Get CAM for selected model and device
    CAM_builder = CAM(model_inst, device='cpu', last_conv_layer=last_conv_layer, fc_layer_name=fc_layer_name, verbose=False)
    pred_cam, _ = CAM_builder.run(instance=window_agg, returned_cam_for_label=1)

    return pred_cam


def plot_detection_probabilities(data):
    # Determine the number of appliances to plot
    num_appliances = len(data)
    appliances = list(data.keys())

    # Create subplots: one row, as many columns as there are appliances
    fig = make_subplots(rows=1, cols=num_appliances, subplot_titles=appliances)

    for i, appliance in enumerate(appliances, start=1):
        appliance_data = data[appliance]
        models = list(appliance_data.keys())
        class_0_probs = [appliance_data[model]['pred_prob'][0] for model in models]
        class_1_probs = [appliance_data[model]['pred_prob'][1] for model in models]

        # Calculating the average probabilities for the ensemble model
        ensemble_class_0_avg = np.mean(class_0_probs)
        ensemble_class_1_avg = np.mean(class_1_probs)

        # Adding the ensemble model to the model list and its probabilities to the lists
        models.append('Ensemble')
        class_0_probs.append(ensemble_class_0_avg)
        class_1_probs.append(ensemble_class_1_avg)

        # Add bars for each class in the subplot
        fig.add_trace(go.Bar(x=models, y=class_0_probs, name='Class 0 Probability', marker_color='indianred'), row=1, col=i)
        fig.add_trace(go.Bar(x=models, y=class_1_probs, name='Class 1 Probability', marker_color='lightsalmon'), row=1, col=i)

    # Update layout once, outside the loop
    fig.update_layout(
        title_text='Probability of Detection for Each Model Including Ensemble (Avg. Probability of each selected models)',
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1, # gap between bars of the same location coordinate.
        height=400, # You can adjust the height based on your needs
        width=1000 # Adjust the width based on the number of appliances or your display requirements
    )

    return fig


def pred_one_window(k, df, window_size, ts_name, appliances, frequency, models):
    window_df = df.iloc[k*window_size: k*window_size + window_size]
    window_agg = window_df['Aggregate']

    pred_dict_all = {}
    for appl in appliances:
        pred_dict_appl      = get_prediction_one_appliance(ts_name, window_agg, appl, frequency, models)
        pred_dict_all[appl] = pred_dict_appl

    return pred_dict_all


def plot_one_window(k, df, window_size, appliances, pred_dict_all):
    window_df = df.iloc[k*window_size: k*window_size + window_size]
    # Plot for 'Aggregate' column for the window
    fig_aggregate_window = go.Figure()
    fig_aggregate_window.add_trace(go.Scatter(x=window_df.index, y=window_df['Aggregate'], mode='lines', name='Aggregate', line=dict(color='royalblue')))
    fig_aggregate_window.update_layout(title='Aggregate Consumption', xaxis_title='Time', yaxis_title='Aggregate Consumption (Watts)', template="plotly")

    # Plot load curve of selected Appliances for the window
    fig_appliances_window = go.Figure()
    for appliance in appliances:
        fig_appliances_window.add_trace(go.Scatter(x=window_df.index, y=window_df[appliance], mode='lines', name=appliance.capitalize()))

    fig_appliances_window.update_layout(title='True Appliance Consumption', xaxis_title='Time', yaxis_title='Appliances Consumption (Watts)', template="plotly")

    # The plots are prepared and can be visualized in a local environment or integrated with Streamlit for dynamic interaction
    # Since the visualization cannot be directy shown here due to the connection issue, please run this code in your local environment
    return fig_aggregate_window,fig_appliances_window,plot_detection_probabilities(pred_dict_all)



def plot_cam(k, df, window_size, appliances, pred_dict_all):
    window_df = df.iloc[k*window_size: k*window_size + window_size]

    fig_cam = make_subplots(rows=len(appliances), cols=1, subplot_titles=[f'CAM {appliance}' for appliance in appliances])

    for appliance in appliances:
        pred_dict_appl = pred_dict_all[appliance]

        for model_name, values in pred_dict_appl.items():
            if values['pred_cam'] is not None:
                # Clip CAM to 0 and set * by predicted label for each model
                cam = np.clip(values['pred_cam'], a_min=0, a_max=None) * values['pred_label']
                fig_cam.add_trace(go.Scatter(x=window_df.index, y=cam, mode='lines', name=f'CAM {model_name}'))

        fig_cam.update_layout(title='CAM', xaxis_title='Time', yaxis_title=f'CAM {appliance}', template="ggplot2")
    return fig_cam

