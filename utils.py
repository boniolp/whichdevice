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

    colcontrol_1, colcontrol_2, colcontrol_3 = st.columns([0.2,0.8,0.2])
    with colcontrol_1:
        if st.button(":rewind: Previous", type="primary"):
            CURRENT_WINDOW -= 1
    with colcontrol_3:
        if st.button("Next :fast_forward:", type="primary"):
            CURRENT_WINDOW += 1
    
    #st.markdown("show TS et prob devices")
    df, window_size = get_time_series_data(ts_name, frequency=frequency, length=length)
    n_win = len(df) // window_size

    with colcontrol_2:
        st.markdown("<p style='text-align: center;'>from {} to {} </p>".format(df.iloc[CURRENT_WINDOW*window_size: (CURRENT_WINDOW+1)*window_size].index[0],df.iloc[CURRENT_WINDOW*window_size: (CURRENT_WINDOW+1)*window_size].index[-1]),unsafe_allow_html=True)
    
    if CURRENT_WINDOW > n_win:
        CURRENT_WINDOW=n_win
    elif CURRENT_WINDOW < 0:
        CURRENT_WINDOW=0
    
    pred_dict_all = pred_one_window(CURRENT_WINDOW, df, window_size, ts_name, appliances, frequency, models)
    fig_ts, fig_app, fig_prob = plot_one_window(CURRENT_WINDOW,  df, window_size, appliances, pred_dict_all)
    
    tab_ts,tab_app = st.tabs(["Aggregated", "Per device"])
    
    with tab_ts:
        st.plotly_chart(fig_ts, use_container_width=True)
    with tab_app:
        st.plotly_chart(fig_app, use_container_width=True)
    
    tab_prob,tab_cam = st.tabs(["Which Appliance?", "When is it used?"])

    with tab_prob:
        st.plotly_chart(fig_prob, use_container_width=True)
    with tab_cam:
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

    dict_color_model = {'ConvNet': 'wheat', 'ResNet': 'coral', 'Inception': 'powderblue', 'TransApp': 'peachpuff', 'Ensemble': 'indianred'}

    # Create subplots: one row, as many columns as there are appliances
    fig = make_subplots(rows=1, cols=num_appliances, subplot_titles=appliances, shared_yaxes=True)

    for i, appliance in enumerate(appliances, start=1):
        appliance_data = data[appliance]
        models = list(appliance_data.keys())
        class_0_probs = [appliance_data[model]['pred_prob'][0] for model in models]
        class_1_probs = [appliance_data[model]['pred_prob'][1] for model in models]
        color_model   = [dict_color_model[model] for model in models]

        # Calculating the average probabilities for the ensemble model
        ensemble_class_0_avg = np.mean(class_0_probs)
        ensemble_class_1_avg = np.mean(class_1_probs)

        # Adding the ensemble model to the model list and its probabilities to the lists
        models.append('Ensemble')
        class_0_probs.append(ensemble_class_0_avg)
        class_1_probs.append(ensemble_class_1_avg)
        color_model.append(dict_color_model['Ensemble'])

        # Add bars for each class in the subplot
        #fig.add_trace(go.Bar(x=models, y=class_0_probs, name='Class 0 Probability', marker_color='indianred'), row=1, col=i)
        fig.add_trace(go.Bar(x=models, y=class_1_probs,  marker_color=color_model), row=1, col=i)

    for axis in fig.layout:
        if axis.startswith('yaxis'):
            fig.layout[axis].update(
                range=[-0.1, 1.1],
                tickmode='array',
                tickvals=[0, 0.5, 1],
                ticktext=['Not Detected', '0.5', 'Detected']
            )

    # Update layout once, outside the loop
    fig.update_layout(
        title_text='Probability of Detection for Each Model',
        barmode='group',
        showlegend=False,
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1, # gap between bars of the same location coordinate.
        height=400, # You can adjust the height based on your needs
        width=1000, # Adjust the width based on the number of appliances or your display requirements
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
    fig_aggregate_window.add_trace(go.Scatter(x=window_df.index, y=window_df['Aggregate'], mode='lines', name='Aggregate', fill='tozeroy', line=dict(color='royalblue')))
    fig_aggregate_window.update_layout(title='Aggregate Consumption', xaxis_title='Time', yaxis_title='Power Consumption (Watts)', template="plotly")

    # Plot load curve of selected Appliances for the window
    fig_appliances_window = go.Figure()
    for appliance in appliances:
        fig_appliances_window.add_trace(go.Scatter(x=window_df.index, y=window_df[appliance], mode='lines', name=appliance.capitalize(), fill='tozeroy'))

    fig_appliances_window.update_layout(title='True Appliance Consumption', 
                                        xaxis_title='Time', 
                                        yaxis_title='Appliances Consumption (Watts)', 
                                        template="plotly",
                                        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2)
                                        )

    return fig_aggregate_window, fig_appliances_window, plot_detection_probabilities(pred_dict_all)


def plot_cam(k, df, window_size, appliances, pred_dict_all):
    window_df = df.iloc[k*window_size: k*window_size + window_size]

    dict_color_model = {'ConvNet': 'wheat', 'ResNet': 'coral', 'Inception': 'powderblue', 'TransApp': 'peachpuff', 'Ensemble': 'indianred'}

    fig_cam = make_subplots(rows=len(appliances), cols=1, subplot_titles=[f'{appliance}' for appliance in appliances], shared_xaxes=True)

    for i, appliance in enumerate(appliances):
        pred_dict_appl = pred_dict_all[appliance]

        for model_name, values in pred_dict_appl.items():
            if values['pred_cam'] is not None:
                # Clip CAM to 0 and set * by predicted label for each model
                cam = np.clip(values['pred_cam'], a_min=0, a_max=None) * values['pred_label']
                fig_cam.add_trace(go.Scatter(x=window_df.index, y=cam, mode='lines', fill='tozeroy', marker=dict(color=dict_color_model[model_name]), name=f'{appliance}: CAM {model_name}'), row=i+1, col=1)

    # Dynamically set the x-axis title for the last subplot
    xaxis_title_dict = {f'xaxis{len(appliances)}_title': 'Time'}

    # Update layout with dynamic x-axis title and general figure properties
    fig_cam.update_layout(title='Class Activation Map to localize appliance pattern', **xaxis_title_dict)

    # Update legend to be at the bottom and horizontal
    fig_cam.update_layout(legend=dict(orientation='h', x=0.5, xanchor='center', y=-1))

    return fig_cam

"""
def plot_cam(k, df, window_size, appliances, pred_dict_all):
    window_df = df.iloc[k*window_size: k*window_size + window_size]

    dict_color_model = {'ConvNet': 'wheat', 'ResNet': 'coral', 'Inception': 'powderblue', 'TransApp': 'peachpuff', 'Ensemble': 'indianred'}

    fig_cam = make_subplots(rows=len(appliances), cols=1, subplot_titles=[f'CAM {appliance}' for appliance in appliances], shared_xaxes=True)

    for i,appliance in enumerate(appliances):
        pred_dict_appl = pred_dict_all[appliance]

        for model_name, values in pred_dict_appl.items():
            if values['pred_cam'] is not None:
                # Clip CAM to 0 and set * by predicted label for each model
                cam = np.clip(values['pred_cam'], a_min=0, a_max=None) * values['pred_label']
                fig_cam.add_trace(go.Scatter(x=window_df.index, y=cam, mode='lines', fill='tozeroy', marker=dict(color=dict_color_model[model_name]), name=f'CAM {model_name}', legendgroup=i+1), row=i+1, col=1)

        fig_cam.update_layout(title='CAM', 
                              xaxis_title='Time', 
                              legend_tracegroupgap=(3 - len(pred_dict_appl))*20+10,
                )
    return fig_cam
"""