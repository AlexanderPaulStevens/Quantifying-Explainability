# Quantifying Explainability in Outcome-Oriented Predictive Process Monitoring
Complementary code to reproduce the work of *Quantifying Explainability in Outcome-Oriented Predictive Process Monitoring*

An overview of the files:


### Exploratory Data Analysis and Feature Selection
- EDA_FeatureSelection

### Hyperoptimalisation of parameters
- Hyperopt_LogitLeafModel
- Hyperopt_MachineLearningModels

### Training of the Machine Learning Models
*Logit Leaf Model (LLM), Generalized Logistic Rule Regression (GLRM) and XGBoost)*
- Experiment_BPIC2017
- Experiment_TF1
- Experiment_BPIC2015

### Training of the Deep Learning Models
*Long short-term memory neural networks (LSTM)*
- LSTM_BPIC2017
- LSTM_TF1
- LSTM_BPIC2015

### Quantitative Metrics

This folder contains the notebook files as listened underneath. Note that for code reproduction of the quantitative metrics, these should be placed outside this folder.
- Experiment_BPIC2017
- Experiment_TF1
- Experiment_BPIC2015
- LSTM_BPIC2017
- LSTM_TF1
- LSTM_BPIC2015

The preprocessing and hyperoptimalisation are derivative work based on the code provided by https://github.com/irhete/predictive-monitoring-benchmark. 
 We would like to thank the authors for the high quality code that allowed to fastly reproduce the provided work.
Secondly, we acknowledgde the work provided by https://github.com/renuka98/interpretable_predictive_processmodel architecture to create the long short-term neural networks with attention layers visualisations.

### Preprocessing files 

- dataset_confs
- DatasetManager
- EncoderFactory

In the Feature Selection file, the original .XES files are used. These can be downloaded from:

- BPIC2017: https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884
- TF1: https://data.4tu.nl/articles/dataset/Road_Traffic_Fine_Management_Process/12683249
- BPIC2015: https://data.4tu.nl/articles/dataset/BPI_Challenge_2015_Municipality_2/12697349/1


Finally, the folders contain additional figures and plots that have not been used in the paper.




