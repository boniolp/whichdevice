# ========================== Texts ========================== #

text_tab_playground = f"""
### Playground:
Explore electrical time series data and select a model to detect and localize appliance patterns."""

text_tab_benchmark = f"""
### Benchmark:
Explore the results of the different classifiers applied to detecting the different appliances for different sampling rates."""


text_description_dataset  = f"""
The electricity consumption data available to test our system come from two different studies and are available publicy online: [UKDALE](https://jack-kelly.com/data/) and [REFIT](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned).
Each dataset is composed of several houses that have been monitored by sensors that record the total main power and appliance-level power for a period of time.

- UKDALE: The UK-DALE dataset contains data from 5 houses in the United Kingdom and includes appliance-level and aggregate load curves sampled at a minimum sampling rate of 6 seconds.
Four houses were recorded for over a year and a half, while the 5th was recorded for 655 days.

- REFIT Dataset: The REFIT dataset contains data from 20 different houses in the United Kingdom that have been monitored with smart meters and multiple sensors. 
This dataset provides aggregate and individual appliance load curves at 8-second sampling intervals. 
"""

text_description_model  = f"""
Appliance detection can be cast as a time series classification problem, where a classifier is trained in a binary supervised manner to detect the presence of an appliance in a consumption series. 
Following results from previous studies, it has been proven that deep learning-based methods are the most accurate and efficient for tackling this task.
Our system provide 4 different time series classifiers to detect appliance:

- ConvNet: Convolutional Neural Network (CNN) is a well-knwon deep learning architecture commonly used in image recognition. 
In this system, we use a variant designed for time series classification. 
The architecture employs stacked 1D convolutional blocks with specific kernel sizes and filters, followed by global average pooling and linear layers for classification.


- ResNet: The Residual Network (ResNet) architecture was originally proposed to adress the gradient vanishing problem in deep CNNs. 
It is also a well-knwon architecture to perform image recognition.
As for the ConvNet, we use an variant proposed for time series classification.
It consists of stacked residual blocks with residual connections, where each block contains 1D convolutional layers with the same kernel sizes and filters. 
A global average pooling, a linear layer, and a softmax activation are used for classification.


- InceptionTime: Inspired by inception-based networks for image classification, InceptionTime is designed for time series classification.
The model employs Inception modules composed of concatenated 1D convolutional layers using different filter sizes.
The outputs are passed through activation and normalization layers; at the end, classification is performed using a global average pooling, followed by a linear layer and softmax activation function.


- TransApp: A recent study proposed TransApp, a deep-learning time series classifier specially designed to detect appliances in electrical consumption series.
The architecture results in an embedding block made of convolutional layers that serves as a features extractor followed by multiple Transformer layers.
In the end, global average pooling, followed by a linear layer and softmax activation function, performs the classification.
As the architecture was originally proposed for detecting appliances in large datasets, we adapted it for our system as a smaller and simplified version by keeping only one Transformer layer after the embedding block.
"""

text_description_explainability = f"""
Identifying the discriminative features that influence the model's decision-making process for classifying each instance is a well-known problem that has been extensively studied in the literature.

- Class Activation Map: For models built on Convolutional Neural Networks (CNNs) architecture, the Class Activation Map (CAM) presents a method for explaining the reasons behind the classification outcomes. 
Originally proposed for explainable image classification, this method highlights the areas within an image that are most influential in the model's prediction for a particular class. 
In recent years, this approach have been applied to time series data and shown promising results and different kinds of data.

- Attention Map Vizualisation : With the recent advent of attention-based models, explainability methods based on the Attention Mechanism have been widely investigated in image recognition.
We adapted this approach to localize discriminative features using the TransApp model.
"""

text_about = f"""
## Interactive detection and localization of appliance patterns in electrical consumption time series

Electricity suppliers have installed millions of smart meters worldwide to improve the management of the smart grid system.
These meters capture detailed time-stamped electricity consumption of the total main power consumed in a house: this recorded signal is hard to analyze as it regroups multiple appliance signatures that run simultaneously.
Making non-expert users (as consumers or sales advisors) understand it has become a major challenge for electricity suppliers.
We propose Deviscope as an interactive solution to facilitate the understanding of electrical data by detecting and localizing individual appliance patterns within recorded time periods.

### How DeviceScope works?
The core of our system is based on a combination of recent works conducted on appliance detection [[1]](https://arxiv.org/abs/2305.10352) [[2]](https://arxiv.org/abs/2401.05381) and explainable classification [[3]](https://arxiv.org/abs/1611.06455) [[4]](https://epfml.github.io/attention-cnn/).
In a nutshell, DeviceScope uses a trained time series classifier to detect ***if*** an appliance is used in a given period of time. If this is the case, a classification explainability method is applied to detect ***when***: it highlights of the region of the consumption series that contributed the most to the classifier's decision.
Contrary to other works in this area (as NILM), the approach is based on a weakly supervised process.
"""


text_info = f"""
### Contributors

* [Adrien Petralia](https://adrienpetralia.github.io/), EDF R&D, Université Paris Cité
* [Paul Boniol](https://boniolp.github.io/), Inria, ENS, PSL University, CNRS
* [Philippe Charpentier](https://www.researchgate.net/profile/Philippe-Charpentier), EDF R&D
* [Themis Palpanas](https://helios2.mi.parisdescartes.fr/~themisp/), Université Paris Cité, IUF

### Acknowledgments
Work supported by EDF R&D and ANRT French program.
"""



# ========================== Lists ========================== #
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
                  'REFIT_House20_2015-06']
