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

frequency_list = ['30 seconds', '1 minutes','10 minutes']
models_list = ['ConvNet','ResNet','Inception','TransApp','Arsenal']
lengths_list = ['6 hours', '12 hours', '1 Day']

def run_playground_frame():
    st.markdown("Here show the time series and CAM")

    frequency = st.selectbox(
        "Choose a sampling rate:", frequency_list
    )
    models = st.multiselect(
        "Choose a model:","ResNet", models_list
    )
    length = st.selectbox(
        "Choose the window length:", lengths_list
    )

def run_benchmark_frame():
    st.markdown("Here show benchmark results")

def run_about_frame():
    st.markdown("Here show info on the models, data and us")
