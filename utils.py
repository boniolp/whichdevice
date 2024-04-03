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
import plotly.express as px
from plotly.subplots import make_subplots

# === Customs import === #
from constants import *
from Models.FCN import FCN
from Models.ResNet import ResNet
from Models.InceptionTime import Inception
from Models.TransAppS import TransAppS
from Helpers.class_activation_map import CAM, AttentionMap

CURRENT_WINDOW=0
    
def run_playground_frame():

    global CURRENT_WINDOW
    
    st.markdown(text_tab_playground)

    col1_1, col1_2, col1_3 = st.columns(3)

    with col1_1:
        ts_name = st.selectbox(
            "Choose a load curve", list_name_ts, index=0
        )
    with col1_2:
        frequency = st.selectbox(
            "Choose a sampling rate:", frequency_list, index=0
        )
    with col1_3:
        length = st.selectbox(
            "Choose the window length:", lengths_list, index=2
        )


    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        appliances1 = st.multiselect(
            "Choose devices:", devices_list, ["Dishwasher", "WashingMachine", "Kettle", "Microwave"]
        )
    with col2_2:
        models = st.multiselect(
            "Choose models:", models_list
        )

    if len(models)>0:
        loc_toggle = st.toggle('Localize appliance patterns')

    colcontrol_1, colcontrol_2, colcontrol_3 = st.columns([0.2,0.8,0.2])
    with colcontrol_1:
        if st.button(":rewind: **Prev.**", type="primary"):
            CURRENT_WINDOW -= 1
    with colcontrol_3:
        if st.button("**Next** :fast_forward:", type="primary"):
            CURRENT_WINDOW += 1
    
    df, window_size = get_time_series_data(ts_name, frequency=frequency, length=length)
    n_win = len(df) // window_size

    if CURRENT_WINDOW > n_win:
        CURRENT_WINDOW=n_win
    elif CURRENT_WINDOW < 0:
        CURRENT_WINDOW=0

    with colcontrol_2:
        st.markdown("<p style='text-align: center;'> <b>from</b> <i>{}</i> <b>to</b> <i>{}</i> </p>".format(df.iloc[CURRENT_WINDOW*window_size: (CURRENT_WINDOW+1)*window_size].index[0],df.iloc[CURRENT_WINDOW*window_size: (CURRENT_WINDOW+1)*window_size].index[-1]),unsafe_allow_html=True)
    
    if len(models)>0:
        pred_dict_all = pred_one_window(CURRENT_WINDOW, df, window_size, ts_name, appliances1, frequency, models)
        if loc_toggle:
            #fig_ts, fig_app, fig_stack = plot_one_window1(CURRENT_WINDOW,  df, window_size, appliances1, pred_dict_all)
            fig_ts, fig_app, fig_stack = plot_one_window3(CURRENT_WINDOW,  df, window_size, appliances1, pred_dict_all)
        else:
            fig_ts, fig_app, fig_stack = plot_one_window2(CURRENT_WINDOW,  df, window_size, appliances1)
            
        fig_prob = plot_detection_probabilities(pred_dict_all)
        
        tab_ts, tab_app = st.tabs(["Aggregated", "Per device"])
        
        with tab_ts:
            st.plotly_chart(fig_ts, use_container_width=True)

            if loc_toggle and len(models)>1:
                st.markdown(f"""**Multiple classifiers are selected:** the heatmap shows the average of predicted pattern localization for selected models.""")
        
        with tab_app:
            on = st.toggle('Stack')
            if on:
                st.plotly_chart(fig_stack, use_container_width=True)
            else:
                st.plotly_chart(fig_app, use_container_width=True)

        if loc_toggle:
            tab_prob, tab_cam, tab_signatures = st.tabs(["Models detection probabilities", "Models patterns localization", "Examples of appliance patterns"])
    
            with tab_prob:
                st.plotly_chart(fig_prob, use_container_width=True)
                if len(models)>1:
                    st.markdown(f"""**Mean Prediction** shows the average of predicted detection probabilities for selected models.""")
            with tab_cam:
                fig_cam = plot_cam(CURRENT_WINDOW, df, window_size, appliances1, pred_dict_all)
                st.plotly_chart(fig_cam, use_container_width=True)
            with tab_signatures:
                fig_sig = plot_signatures(appliances1, frequency)
                st.plotly_chart(fig_sig, use_container_width=True)
        else:
            tab_prob, tab_signatures = st.tabs(["Models detection probabilities", "Examples of appliance patterns"])
    
            with tab_prob:
                st.plotly_chart(fig_prob, use_container_width=True)
                if len(models)>1:
                    st.markdown(f"""**Mean Prediction** shows the average of predicted detection probabilities for selected models.""")
            with tab_signatures:
                fig_sig = plot_signatures(appliances1, frequency)
                st.plotly_chart(fig_sig, use_container_width=True)
    else:
        fig_ts, fig_app, fig_stack = plot_one_window2(CURRENT_WINDOW,  df, window_size, appliances1)

        tab_ts, tab_app = st.tabs(["Aggregated", "Per device"])

        with tab_ts:
            st.plotly_chart(fig_ts, use_container_width=True)
        
        with tab_app:
            on = st.toggle('Stack')
            if on:
                st.plotly_chart(fig_stack, use_container_width=True)
            else:
                st.plotly_chart(fig_app, use_container_width=True)

        st.markdown(f"""Look at the example patterns of selected devices below and try to identify them in the aggregate main power!""")
        
        fig_sig = plot_signatures(appliances1, frequency)

        st.plotly_chart(fig_sig, use_container_width=True)
        
            
    

def run_benchmark_frame():
    st.markdown(text_tab_benchmark)

    col1, col2 = st.columns(2)

    with col1:
        measure = st.selectbox(
            "Choose metric", measures_list, index=0
        )
    with col2:
        dataset = st.selectbox(
            "Choose dataset", dataset_list, index=0
        )

    #st.markdown("#### Overall results")

    fig1 = plot_benchmark_figures1(measure, dataset)
    fig2 = plot_benchmark_figures2(measure, dataset)
    fig3 = plot_benchmark_figures3(measure, dataset)

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

    #st.markdown("#### Explore the influence of the sampling rate on the detection performance for selected appliance(s).")

    appliances2 = st.multiselect(
        "Select devices:", devices_list, ["Dishwasher", "WashingMachine", "Kettle", "Microwave"]
    )

    fig_benchmark = plot_benchmark_figures4(appliances2, measure, dataset)
    st.plotly_chart(fig_benchmark, use_container_width=True)
    
    

def run_about_frame():

    st.markdown(text_about)

    st.image("Figures/DeviceScopePipelineGithub.png", caption="Proposed appliance detection pipeline.")
    
    with st.expander(f"""### Appliance detection as a time series classification problem"""):
        st.markdown(text_description_model)

    with st.expander(f"""### Explainable classification to localize appliance patterns"""):
        st.write(text_description_explainability)

    st.markdown("""### Smart meters datasets""")
    st.markdown(text_description_dataset)

    st.markdown(text_info)

def plot_benchmark_figures1(name_measure, dataset):
    table = pd.read_csv(os.getcwd()+'/TableResults/Results.gzip', compression='gzip')
    if dataset != 'All':
        table = table.loc[table['Dataset'] == dataset]

    dict_measure = {'Accuracy': 'Acc', 'Balanced Accuracy': 'Acc_Balanced', 'F1 Macro': 'F1_Macro'}
    measure = dict_measure[name_measure]

    table = table[['Models'] + [measure]].groupby(['Models'], as_index=False).mean()

    table = table.sort_values(measure)

    dict_color_model = {'ConvNet': 'wheat', 'ResNet': 'coral', 'Inception': 'powderblue', 'TransAppS': 'indianred', 'Ensemble': 'peachpuff'}


    min_val = table[measure].values.flatten().min()
    fig = px.bar(table, x='Models', y=measure, labels={measure: name_measure},
                 color='Models', 
                 color_discrete_map=dict_color_model, 
                 range_y=[min(0.5, round(min_val-0.1)), 1],
                 height=400,
                 title='Overall models performance for selected dataset')
    
    return fig

def plot_benchmark_figures2(name_measure, dataset):
    table = pd.read_csv(os.getcwd()+'/TableResults/Results.gzip', compression='gzip')
    if dataset != 'All':
        table = table.loc[table['Dataset'] == dataset]

    dict_measure = {'Accuracy': 'Acc', 'Balanced Accuracy': 'Acc_Balanced', 'F1 Macro': 'F1_Macro'}
    measure = dict_measure[name_measure]

    table = table[['Appliance', 'Models']+[measure]].groupby(['Appliance', 'Models'], as_index=False).mean()

    # Assuming grouped_df is your DataFrame after grouping and sorting
    table = table.sort_values(['Models', 'Appliance'])

    table['Appliance'] = table['Appliance'].astype('category')

    dict_color_appliance = {'WashingMachine': 'teal', 'Dishwasher': 'skyblue', 'Kettle': 'orange', 'Microwave': 'grey'}

    min_val = table[measure].values.flatten().min()
    # Create the grouped bar plot
    fig = px.bar(table, 
                x='Models', 
                y=measure, labels={measure: name_measure},
                color='Appliance',
                color_discrete_map=dict_color_appliance,
                barmode='group',
                range_y=[min(0.5, round(min_val-0.1)), 1], 
                height=400,
                title='Models performance for each appliance for selected dataset')
    
    return fig


def plot_benchmark_figures3(name_measure, dataset):
    table = pd.read_csv(os.getcwd()+'/TableResults/Results.gzip', compression='gzip')
    if dataset != 'All':
        table = table.loc[table['Dataset'] == dataset]

    dict_measure = {'Accuracy': 'Acc', 'Balanced Accuracy': 'Acc_Balanced', 'F1 Macro': 'F1_Macro'}
    measure = dict_measure[name_measure]
    
    table = table[['SamplingRate', 'Models']+[measure]].groupby(['SamplingRate', 'Models'], as_index=False).mean()

    sampling_order = ['30s', '1T', '10T']  # Define the logical order
    table['SamplingRate_order'] = pd.Categorical(table['SamplingRate'], categories=sampling_order, ordered=True)

    table = table.sort_values(['SamplingRate_order', 'Models'])

    table['SamplingRate'] = table['SamplingRate'].astype('category')

    dict_color_sp = {'30s': 'rgb(211, 211, 211)', '1T': 'rgb(128, 128, 128)', '10T': 'black'}

    min_val = table[measure].values.flatten().min()

    fig = px.bar(table, 
                x='Models', 
                y=measure, labels={measure: name_measure},
                color='SamplingRate',
                color_discrete_map=dict_color_sp,
                barmode='group',
                range_y=[min(0.5, round(min_val-0.1)), 1], 
                height=400,
                title='Models performance for each sampling rate for selected dataset')
    
    return fig


def plot_benchmark_figures4(appliances, measure, dataset):
    df = pd.read_csv(os.getcwd()+'/TableResults/Results.gzip', compression='gzip')
    sampling_rates = df['SamplingRate'].unique()

    if dataset != 'All':
        df = df.loc[df['Dataset'] == dataset]

    dict_color_model = {'ConvNet': 'wheat', 'ResNet': 'coral', 'Inception': 'powderblue', 'TransAppS': 'indianred', 'Ensemble': 'peachpuff'}
    dict_measure = {'Accuracy': 'Acc', 'Balanced Accuracy': 'Acc_Balanced', 'F1 Macro': 'F1_Macro'}

    # Create subplots: one column for each appliance, shared y-axis
    fig = make_subplots(rows=1, cols=len(appliances), shared_yaxes=True, subplot_titles=[f"{appliance}" for appliance in appliances])

    legend_added = []

    added_models = set() 

    for j, appliance in enumerate(appliances, start=1):
        for model_name in ['ConvNet', 'ResNet', 'Inception', 'TransAppS']:
            accuracies = [df[(df['Appliance'] == appliance) & (df['SamplingRate'] == sr) & (df['Models'] == model_name)][dict_measure[measure]].values[0] for sr in sampling_rates]

            show_legend = model_name not in added_models  
            added_models.add(model_name)  

            fig.add_trace(go.Scatter(x=sampling_rates, y=accuracies, mode='lines+markers',
                                    name=model_name, marker_color=dict_color_model[model_name],
                                    marker=dict(size=10), showlegend=show_legend,
                                    legendgroup=model_name),
                          row=1, col=j)
            
            if show_legend:
                legend_added.append(model_name)

    # Update y-axes for each subplot to have the range [0, 1]
    for j in range(1, len(appliances) + 1):
        fig.update_yaxes(range=[0, 1.05], row=1, col=j)
        fig.update_xaxes(title_text="Sampling Rate", row=1, col=j)

    fig.update_layout(
        title='Sampling rate influence on the detection performance of each classifier for selected appliance(s)',
        xaxis_title="Sampling Rate",
        yaxis_title=measure,
        legend_title="Model",
        font=dict(size=13)
    )

    return fig


def get_model_instance(model_name, win_size):
    # Load instance according to selected model
    if model_name=='ConvNet':
        model_inst = FCN()
    elif model_name=='ResNet':
        model_inst = ResNet()
    elif model_name=='Inception':
        model_inst = Inception()
    elif model_name=='TransAppS':
        model_inst = TransAppS(c_in=1, window_size=win_size,  store_att=True)
    else:
        raise ValueError(f'Model {model_name} unknown.')

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
    df = pd.read_csv(os.getcwd()+f'/Data/{ts_name}.gzip', compression='gzip', parse_dates=['Time']).set_index('Time')
    
    # Resample to choosen frequency (if > 30s)
    if pd_freq!='30s':
        df = df.resample(pd_freq).mean()

    return df, window_size


def get_prediction_one_appliance(ts_name, window_agg, appliance, frequency, model_list):
    dict_freq  = {'30 seconds': '30s', '1 minutes': '1T', '10 minutes': '10T'}
    dic_win    = {'30 seconds': 2880,  '1 minutes': 1440, '10 minutes':  144}
    sampling_rate = dict_freq[frequency]

    window_agg  = torch.Tensor(window_agg).unsqueeze(0).unsqueeze(0)

    pred_dict = {}
        
    for model_name in model_list:
        # Get model instance
        model_inst = get_model_instance(model_name, dic_win[frequency])
        # Load compressed model
        path_model = os.getcwd()+f'/TrainedModels/{get_dataset_name(ts_name)}/{sampling_rate}/{appliance}/{model_name}.pt.xz'
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

        # Predict CAM or AttMap
        #if model_name in ['ConvNet', 'ResNet', 'Inception']:
        pred_cam = get_cam(window_agg, model_name, model_inst, sampling_rate)

        # Update pred_dict
        pred_dict[model_name] = {'pred_prob': pred_prob, 'pred_label': pred_label, 'pred_cam': pred_cam}

    return pred_dict

def get_cam(window_agg, model_name, model_inst, sampling_rate):

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
    elif model_name=='TransAppS':
        n_encoder_layers = 1

    # Get CAM for selected model and device
    if model_name=='TransAppS':
        CAM_builder = AttentionMap(model_inst, device='cpu', n_encoder_layers=n_encoder_layers, merge_channels_att='sum', head_att='sum')
        pred_cam, _ = CAM_builder.run(instance=window_agg, return_att_for='all')
        dict_conv  = {'30s': 20, '1T': 10, '10T':5}
        pred_cam = np.convolve(pred_cam, np.ones(dict_conv[sampling_rate]), mode='same')
        pred_cam = scale_cam_inst(pred_cam)
    else:
        CAM_builder = CAM(model_inst, device='cpu', last_conv_layer=last_conv_layer, fc_layer_name=fc_layer_name, verbose=False)
        pred_cam, _ = CAM_builder.run(instance=window_agg, returned_cam_for_label=1)
        pred_cam = scale_cam_inst(pred_cam)

    return pred_cam


def pred_one_window(k, df, window_size, ts_name, appliances, frequency, models):
    window_df = df.iloc[k*window_size: k*window_size + window_size]
    window_agg = window_df['Aggregate']

    pred_dict_all = {}
    for appl in appliances:
        pred_dict_appl      = get_prediction_one_appliance(ts_name, window_agg, appl, frequency, models)
        pred_dict_all[appl] = pred_dict_appl

    return pred_dict_all


def plot_one_window1(k, df, window_size, appliances, pred_dict_all):
    window_df = df.iloc[k*window_size: k*window_size + window_size]
    dict_color_appliance = {'WashingMachine': 'teal', 'Dishwasher': 'skyblue', 'Kettle': 'orange', 'Microwave': 'grey'}
    
    # Create subplots with 2 rows, shared x-axis
    size_cam = 0.1 * (len(appliances)+1)

    fig_agg          = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[1-size_cam, size_cam])
    fig_appl         = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[1-size_cam, size_cam])
    fig_appl_stacked = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[1-size_cam, size_cam])
    
    # Aggregate plot
    fig_agg.add_trace(go.Scatter(x=window_df.index, y=window_df['Aggregate'], mode='lines', name='Aggregate', fill='tozeroy', line=dict(color='royalblue')),
                  row=1, col=1)
    
    # Stacked CAM heatmap calculations
    z = []
    for appl in appliances:
        fig_appl.add_trace(go.Scatter(x=window_df.index, y=window_df[appl], mode='lines', name=appl.capitalize(), marker_color=dict_color_appliance[appl],  fill='tozeroy'))
        fig_appl_stacked.add_trace(go.Scatter(x=window_df.index, y=window_df[appl], mode='lines', line=dict(width=0), marker_color=dict_color_appliance[appl], name=appl.capitalize(), stackgroup='one'))

        stacked_cam = None
        dict_pred = pred_dict_all[appl]

        k = 0
        for name_model, dict_model in dict_pred.items():
            if dict_model['pred_cam'] is not None:
                # Aggregate CAMs from different models
                if dict_model['pred_label'] < 1:
                    if name_model == 'TransAppS':
                        tmp_cam = dict_model['pred_cam'] * 0
                    else:
                        tmp_cam = dict_model['pred_cam'] * dict_model['pred_prob'][1]
                else:
                    tmp_cam = dict_model['pred_cam']

                stacked_cam = stacked_cam + tmp_cam if stacked_cam is not None else tmp_cam
                k += 1
        
        # Clip values and ensure it's an array with the same length as window_agg
        stacked_cam = np.clip(stacked_cam/k, a_min=0, a_max=None) if stacked_cam is not None else np.zeros(len(window_df['Aggregate']))
        z.append(stacked_cam)
    
    # Heatmap for stacked CAM
    fig_agg.add_trace(go.Heatmap(z=z, x=window_df.index, y=appliances, colorscale='RdBu_r', showscale=False, zmin=0, zmax=1), row=2, col=1)
    fig_appl.add_trace(go.Heatmap(z=z, x=window_df.index, y=appliances, colorscale='RdBu_r', showscale=False, zmin=0, zmax=1), row=2, col=1)
    fig_appl_stacked.add_trace(go.Heatmap(z=z, x=window_df.index, y=appliances, colorscale='RdBu_r', showscale=False, zmin=0, zmax=1), row=2, col=1)
    
    # Update layout for the combined figure
    fig_agg.update_layout(
        title='Aggregate power consumption and predicted appliance localization',
        xaxis2_title='Time',
        height=500,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40)
    )

    fig_appl.update_layout(
        title='Individual appliance power consumption compared to predicted appliance localization',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2),
        xaxis2_title='Time',
        height=500,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40)
    )

    fig_appl_stacked.update_layout(
        title='Individual appliance power consumption (stacked) compared to predicted appliance localization',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2),
        xaxis2_title='Time',
        height=500,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40)
    )
    
    # Update y-axis for the aggregate consumption plot
    fig_agg.update_yaxes(title_text='Power (Watts)', row=1, col=1, range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])
    fig_appl.update_yaxes(title_text='Power (Watts)', row=1, col=1, range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])
    fig_appl_stacked.update_yaxes(title_text='Power (Watts)', row=1, col=1, range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])
    
    # Update y-axis for the heatmap
    fig_agg.update_yaxes(tickmode='array', tickvals=list(appliances), ticktext=appliances, row=2, col=1, tickangle=-45)
    fig_appl.update_yaxes(tickmode='array', tickvals=list(appliances), ticktext=appliances, row=2, col=1, tickangle=-45)
    fig_appl_stacked.update_yaxes(tickmode='array', tickvals=list(appliances), ticktext=appliances, row=2, col=1, tickangle=-45)

    return fig_agg, fig_appl, fig_appl_stacked


def plot_one_window2(k, df, window_size, appliances):
    window_df = df.iloc[k*window_size: k*window_size + window_size]
    dict_color_appliance = {'WashingMachine': 'teal', 'Dishwasher': 'skyblue', 'Kettle': 'orange', 'Microwave': 'grey'}

    fig_agg          = go.Figure()
    fig_appl         = go.Figure()
    fig_appl_stacked = go.Figure()
    
    # Aggregate plot
    fig_agg.add_trace(go.Scatter(x=window_df.index, y=window_df['Aggregate'], mode='lines', name='Aggregate', fill='tozeroy', line=dict(color='royalblue')))
    
    for appl in appliances:
        fig_appl.add_trace(go.Scatter(x=window_df.index, y=window_df[appl], mode='lines', name=appl.capitalize(), marker_color=dict_color_appliance[appl], fill='tozeroy'))
        fig_appl_stacked.add_trace(go.Scatter(x=window_df.index, y=window_df[appl], mode='lines', line=dict(width=0), marker_color=dict_color_appliance[appl], name=appl.capitalize(), stackgroup='one'))
    
    # Update layout for the combined figure
    fig_agg.update_layout(
        title='Aggregate power consumption',
        xaxis_title='Time',
        height=300,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40)
    )

    fig_appl.update_layout(
        title='Individual appliance power consumption',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.5),
        xaxis_title='Time',
        height=300,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40)
    )

    fig_appl_stacked.update_layout(
        title='Individual appliance power consumption (stacked)',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2),
        xaxis_title='Time',
        height=300,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40)
    )
    
    # Update y-axis for the aggregate consumption plot
    fig_agg.update_yaxes(title_text='Power (Watts)', range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])
    fig_appl.update_yaxes(title_text='Power (Watts)', range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])
    fig_appl_stacked.update_yaxes(title_text='Power (Watts)', range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])

    return fig_agg, fig_appl, fig_appl_stacked


def plot_one_window3(k, df, window_size, appliances, pred_dict_all):
    window_df = df.iloc[k*window_size: k*window_size + window_size]
    dict_color_appliance = {'WashingMachine': 'teal', 'Dishwasher': 'skyblue', 'Kettle': 'orange', 'Microwave': 'grey'}
    
    # Create subplots with 2 rows, shared x-axis
    list_row_heights = [0.6] + [0.4/len(appliances) for _ in range(len(appliances))]

    fig_agg          = make_subplots(rows=len(appliances)+1, cols=1, 
                                     shared_xaxes=True, vertical_spacing=0.1, row_heights=list_row_heights,
                                     subplot_titles=[""]+appliances)
    fig_appl         = make_subplots(rows=len(appliances)+1, cols=1, shared_xaxes=True, 
                                     vertical_spacing=0.1, row_heights=list_row_heights,
                                     subplot_titles=[""]+appliances)
    fig_appl_stacked = make_subplots(rows=len(appliances)+1, cols=1, shared_xaxes=True, 
                                     vertical_spacing=0.1, row_heights=list_row_heights,
                                     subplot_titles=[""]+appliances)
    
    # Aggregate plot
    fig_agg.add_trace(go.Scatter(x=window_df.index, y=window_df['Aggregate'], mode='lines', name='Aggregate', fill='tozeroy', line=dict(color='royalblue')),
                      row=1, col=1)
    
    # Stacked CAM calculations
    for z, appl in enumerate(appliances, start=1):

        fig_appl.add_trace(go.Scatter(x=window_df.index, y=window_df[appl], mode='lines', name=appl.capitalize(), marker_color=dict_color_appliance[appl],  fill='tozeroy'))
        fig_appl_stacked.add_trace(go.Scatter(x=window_df.index, y=window_df[appl], mode='lines', line=dict(width=0), marker_color=dict_color_appliance[appl], name=appl.capitalize(), stackgroup='one'))

        stacked_cam = None
        dict_pred = pred_dict_all[appl]

        k = 0
        for name_model, dict_model in dict_pred.items():
            if dict_model['pred_cam'] is not None:
                # Aggregate CAMs from different models
                if dict_model['pred_label'] < 1:
                    tmp_cam = dict_model['pred_cam'] * 0
                    #if name_model == 'TransAppS':
                    #    tmp_cam = dict_model['pred_cam'] * 0
                    #else:
                    #    tmp_cam = dict_model['pred_cam'] * dict_model['pred_prob'][1]
                else:
                    tmp_cam = dict_model['pred_cam']

                stacked_cam = stacked_cam + tmp_cam if stacked_cam is not None else tmp_cam
                k += 1
        
        # Clip values and ensure it's an array with the same length as window_agg
        stacked_cam = np.clip(stacked_cam/k, a_min=0, a_max=None) if stacked_cam is not None else np.zeros(len(window_df['Aggregate']))

        if appl=='WashingMachine' or appl=='Dishwasher':
            w=30
        elif appl=='Kettle':
            w=3
        else:
            w=3
        #stacked_cam = np.convolve(stacked_cam, np.ones(w), 'same') / w
    
        # Stacked CAM
        fig_agg.add_trace(go.Scatter(x=window_df.index, y=stacked_cam, mode='lines', showlegend=False, name=appl.capitalize(), marker_color=dict_color_appliance[appl], fill='tozeroy'), row=1+z, col=1)
        fig_appl.add_trace(go.Scatter(x=window_df.index, y=stacked_cam, mode='lines', showlegend=False,  name=appl.capitalize(), marker_color=dict_color_appliance[appl],  fill='tozeroy'), row=1+z, col=1)
        fig_appl_stacked.add_trace(go.Scatter(x=window_df.index, y=stacked_cam, mode='lines', showlegend=False,  name=appl.capitalize(), marker_color=dict_color_appliance[appl],  fill='tozeroy'), row=1+z, col=1)
        

        # Example modification: Iterate over stacked_cam to identify and draw rectangles
        color = dict_color_appliance[appl]  # Get color for the current appliance
        start_idx = None  # Start index of the active segment

        threshold = np.mean(stacked_cam)

        for i, value in enumerate(stacked_cam):
            if value > threshold and start_idx is None:  # CAM becomes active
                start_idx = i
            elif value <= threshold and start_idx is not None:  # End of an active segment
                # Add shape for the active segment
                fig_agg.add_shape(
                    type="rect",
                    x0=window_df.index[start_idx],  # Convert index to x-value as needed
                    y0=0,
                    x1=window_df.index[i],
                    y1=max(3000, np.max(window_df['Aggregate'].values) + 50),
                    line=dict(width=0),
                    fillcolor=color,
                    opacity=0.3,  # Adjust for desired transparency
                    layer="below",
                    row=1, col=1
                )
                start_idx = None  # Reset for next segment

        # Check if there's an active segment until the end
        if start_idx is not None:
            fig_agg.add_shape(
                type="rect",
                x0=window_df.index[start_idx],
                y0=0,
                x1=window_df.index[-1],
                y1=max(3000, np.max(window_df['Aggregate'].values) + 50),
                line=dict(width=0),
                fillcolor=color,
                opacity=0.3,
                layer="below",
                row=1, col=1
            )

    # Update layout for the combined figure
    xaxis_title_dict = {f'xaxis{len(appliances)+1}_title': 'Time'}
    fig_agg.update_layout(
        title='Aggregate power consumption and predicted appliance localization',
        showlegend=False,
        height=500,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40),
        **xaxis_title_dict
    )
    
    fig_appl.update_layout(
        title='Individual appliance power consumption compared to predicted appliance localization',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2),
        height=500,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40),
        **xaxis_title_dict
    )

    fig_appl_stacked.update_layout(
        title='Individual appliance power consumption compared to predicted appliance localization',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2),
        height=500,
        width=1000,
        margin=dict(l=100, r=20, t=30, b=40),
        **xaxis_title_dict
    )
    
    fig_agg.update_annotations(font=dict(family="Helvetica", size=15))
    fig_appl.update_annotations(font=dict(family="Helvetica", size=15))
    fig_appl_stacked.update_annotations(font=dict(family="Helvetica", size=15))

    fig_agg.update_yaxes(title_text='Power (Watts)', row=1, col=1, range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])
    fig_appl.update_yaxes(title_text='Power (Watts)', row=1, col=1, range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])
    fig_appl_stacked.update_yaxes(title_text='Power (Watts)', row=1, col=1, range=[0, max(3000, np.max(window_df['Aggregate'].values) + 50)])
    
    # Update y-axis for the heatmap
    for z, appl in enumerate(appliances, start=2):
        fig_agg.update_yaxes(row=z, col=1, range=[0, 1], visible=False, showticklabels=False)
        fig_appl.update_yaxes(row=z, col=1, range=[0, 1], visible=False, showticklabels=False)
        fig_appl_stacked.update_yaxes(row=z, col=1, range=[0, 1], visible=False, showticklabels=False)
    #fig_agg.update_yaxes(tickmode='array', tickvals=list(appliances), ticktext=appliances, row=2, col=1, tickangle=-45)
    #fig_appl.update_yaxes(tickmode='array', tickvals=list(appliances), ticktext=appliances, row=2, col=1, tickangle=-45)
    #fig_appl_stacked.update_yaxes(tickmode='array', tickvals=list(appliances), ticktext=appliances, row=2, col=1, tickangle=-45)

    return fig_agg, fig_appl, fig_appl_stacked


def plot_detection_probabilities(data):
    # Determine the number of appliances to plot
    num_appliances = len(data)
    appliances = list(data.keys())

    dict_color_model = {'ConvNet': 'wheat', 'ResNet': 'coral', 'Inception': 'powderblue', 'TransAppS': 'indianred', 'Ensemble': 'peachpuff'}

    # Create subplots: one row, as many columns as there are appliances
    fig = make_subplots(rows=1, cols=num_appliances, subplot_titles=appliances, shared_yaxes=True)

    for i, appliance in enumerate(appliances, start=1):
        appliance_data = data[appliance]
        models = list(appliance_data.keys())
        #class_0_probs = [appliance_data[model]['pred_prob'][0] for model in models]
        class_1_probs = [appliance_data[model]['pred_prob'][1] for model in models]
        color_model   = [dict_color_model[model] for model in models]

        # Calculating the average probabilities for the ensemble model
        #ensemble_class_0_avg = np.mean(class_0_probs)
        ensemble_class_1_avg = np.mean(class_1_probs)

        # Adding the ensemble model to the model list only if multiple selected models
        if len(models)>1:
            models.append('Mean Prediciton')
            #class_0_probs.append(ensemble_class_0_avg)
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
        title_text='Probability of detection for each classifier',
        barmode='group',
        showlegend=False,
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1, # gap between bars of the same location coordinate.
        height=400, # You can adjust the height based on your needs
        width=1000, # Adjust the width based on the number of appliances or your display requirements
    )

    return fig


def plot_cam(k, df, window_size, appliances, pred_dict_all):
    window_df = df.iloc[k*window_size: k*window_size + window_size]

    dict_color_model = {'ConvNet': 'wheat', 'ResNet': 'coral', 'Inception': 'powderblue', 'TransAppS': 'indianred', 'Ensemble': 'peachpuff'}

    fig_cam = make_subplots(rows=len(appliances), cols=1, subplot_titles=[f'{appliance}' for appliance in appliances], shared_xaxes=True)

    added_models = set()  # Track which models have been added to figure for legend purposes

    for i, appliance in enumerate(appliances):
        pred_dict_appl = pred_dict_all[appliance]

        for model_name, values in pred_dict_appl.items():
            if values['pred_cam'] is not None:
                cam = np.clip(values['pred_cam'], a_min=0, a_max=None) * values['pred_label']

                show_legend = model_name not in added_models  # Show legend only if model hasn't been added
                added_models.add(model_name)  # Mark model as added

                fig_cam.add_trace(go.Scatter(x=window_df.index, y=cam, mode='lines', fill='tozeroy',
                                             marker=dict(color=dict_color_model[model_name]),
                                             name='AttMap TransAppS' if model_name=='TransAppS' else f'CAM {model_name}',
                                             legendgroup=model_name,  # Assign legend group
                                             showlegend=show_legend),
                                  row=i+1, col=1)
        
        fig_cam.update_yaxes(range=[0, 1], row=i+1, col=1)

    xaxis_title_dict = {f'xaxis{len(appliances)}_title': 'Time'}
    fig_cam.update_layout(title='Detail of explainable pattern localization for each classifier', **xaxis_title_dict)
    fig_cam.update_layout(legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.3),
                          height=500,
                          width=1000,
                          margin=dict(l=110, r=20, t=100, b=50))

    return fig_cam

    


def scale_cam_inst(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = 2 * (arr - min_val) / (max_val - min_val) - 1

    return scaled_arr



def plot_signatures(appliances, frequency):
    fig = make_subplots(rows=1, cols=len(appliances), subplot_titles=[f'{appliance}' for appliance in appliances], shared_yaxes=True)
    dict_freq  = {'30 seconds': '30s', '1 minutes': '1T', '10 minutes': '10T'}
    dict_color_appliance = {'WashingMachine': 'teal', 'Dishwasher': 'skyblue', 'Kettle': 'orange', 'Microwave': 'grey'}
    sampling_rate = dict_freq[frequency]

    for i, appliance in enumerate(appliances, start=1):
        print(appliance)
        signature = pd.read_csv(os.getcwd()+f'/Data/example_{appliance}.gzip', parse_dates=['Time'], compression='gzip').set_index('Time')
        signature = signature.resample(sampling_rate).mean()

        fig.add_trace(go.Scatter(x=signature.index, y=signature[appliance], 
                                 marker_color=dict_color_appliance[appliance], 
                                 mode='lines', fill='tozeroy'),
                          row=1, col=i)
        
      # Update y-axes for each subplot to have the range [0, 1]
    for j in range(1, len(appliances) + 1):
        fig.update_xaxes(title_text="Time", row=1, col=j)
        
    fig.update_layout(title='Example of signature for each appliance (according to selected sampling rate)', 
                      yaxis_title='Power (Watts)', 
                      showlegend=False,
                      height=400, 
                      margin=dict(l=100, r=30, t=70, b=40),
                      yaxis_range=[0, 6000]
                    )

    return fig
