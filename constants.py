frequency_list = ['30 seconds', '1 minutes','10 minutes']
models_list    = ['ConvNet', 'ResNet', 'Inception', 'TransAppS']
lengths_list   = ['6 hours', '12 hours', '1 Day']
devices_list   = ['WashingMachine', 'Dishwasher', 'Microwave', 'Kettle']
measures_list  = ['Accuracy', 'Balanced Accuracy', 'F1 Macro']
dataset_list   = ['All', 'UKDALE', 'REFIT']

list_name_ts   = ['UKDALE_House2_2013-05', 
                    'UKDALE_House2_2013-06', 
                    'UKDALE_House2_2013-07', 
                    'UKDALE_House2_2013-08', 
                    'UKDALE_House2_2013-09', 
                    'UKDALE_House2_2013-10',
                    'REFIT_House2_2013-09',
                    'REFIT_House2_2013-10',
                    'REFIT_House2_2013-11',
                    'REFIT_House2_2013-12',
                    'REFIT_House2_2014-01',
                    'REFIT_House2_2014-02',
                    'REFIT_House2_2014-03',
                    'REFIT_House2_2014-04',
                    'REFIT_House2_2014-05',
                    'REFIT_House2_2014-06',
                    'REFIT_House2_2014-07',
                    'REFIT_House2_2014-08',
                    'REFIT_House2_2014-09',
                    'REFIT_House2_2014-10',
                    'REFIT_House2_2014-11',
                    'REFIT_House2_2014-12',
                    'REFIT_House2_2015-01',
                    'REFIT_House2_2015-02',
                    'REFIT_House2_2015-03',
                    'REFIT_House2_2015-04',
                    'REFIT_House2_2015-05',
                    'REFIT_House20_2014-03',
                    'REFIT_House20_2014-04',
                    'REFIT_House20_2014-05',
                    'REFIT_House20_2014-06',
                    'REFIT_House20_2014-07',
                    'REFIT_House20_2014-08',
                    'REFIT_House20_2014-09',
                    'REFIT_House20_2014-10',
                    'REFIT_House20_2014-11',
                    'REFIT_House20_2014-12',
                    'REFIT_House20_2015-01',
                    'REFIT_House20_2015-02',
                    'REFIT_House20_2015-03',
                    'REFIT_House20_2015-04',
                    'REFIT_House20_2015-05',
                    'REFIT_House20_2015-06'
                ]

text_description_dataset  = f"""
Two different datasets of electricity consumption were used in for this demonstraion: UKDALE and REFIT.
Each dataset is composed of several houses that have been monitored by sensors that record the total main power and appliance-level power for a period of time. 

- UKDALE: The UK-DALE dataset contains data from 5 houses in the United Kingdom and includes appliance-level load curves sampled every 6 seconds, as well as the whole-house aggregate data series sampled at 16kHz. 
Four houses were recorded for over a year and a half, while the 5th was recorded for 655 days.


- REFIT Dataset: The REFIT project (Personalised Retrofit Decision Support Tools for UK Homes using Smart Home Technology) ran between 2013 and 2015. 
During this period, 20 houses in the United Kingdom were recorded after being monitored with smart meters and multiple sensors. 
This dataset provides aggregate and individual appliance load curves at 8-second sampling intervals. 
"""

text_description_model  = f"""
Appliance detection can be cast as a time series classification problem.
To do so, a classifier is trained to detect the presence of an appliance in a consumption time series as a supervised binary classification problem (yes/no). 
The 4 methods used in this demonstration have been selected based on their performance in previous studies for the appliance detection tasks and their ability to be combined to explainability approaches.

- ConvNet: Convolutional Neural Network (CNN) is a deep learning architecture commonly used in image recognition. 
The ConvNet variant, we use in this study employs stacked convolutional blocks with specific kernel sizes and filters, followed by global average pooling and linear layers for classification.


- ResNet: The Residual Network (ResNet) architecture addresses the gradient vanishing problem in large CNNs. 
The adaptation for time series classification consists of stacked residual blocks with residual connections, where each block contains 1D convolutional layers with the same kernel sizes and filters. 
A global average pooling, a linear layer, and a softmax activation are used for classification.


-InceptionTime: Inspired by inception-based networks for image classification, InceptionTime is designed for time series classification.
It employs Inception modules composed of concatenated convolutional layers using different filter sizes.
The outputs are passed through activation and normalization layers; at the end, classification is performed using a global average pooling, followed by a linear layer and softmax activation function.


-TransApp: In a recent study, the authors propose a Convolution-Transformer-based architecture to detect appliances in long and variable length consumption series.
The architecture is a combination of a dilated convolution block followed by multiple Transformer layers.
We adapt the proposed architecture to our problem as a smaller and simplified architecture by keeping only one Transformer layer after the convolution embedding block.
"""