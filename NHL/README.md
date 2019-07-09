# NHL

This project was created partially for a meeting of the TRIUMF Data Science group, but mostly out of interest in the NHL! 
The TRIUMF group wanted to learn about Recurrent Neural Networks, and the time-series data of player stats from year to year seemed like a place 
that an RNN could be incorporated, so I built the project to include the TRIUMF lecture as a result. 

It is a simple introduction to using Keras to produce recurrent neural networks. 
Specifically, the functional model is used with LSTM layers.
It is also an introduction to using Keras callbacks and the Tensorboard Visualizations

It uses a dataset that comes in part from CapFriendly.com and partially from Evolving-Hockey.com 

## Goals of the Project:

1. Wrangle and scrape the data. Produce usable dataset files.

2. Visualize the dataset.

3. Produce and train neural networks in Keras. 

4. Assess the accuracy.

5. Become familiar with Tensorboard.

6. Make predictions for new up and coming contracts!

## Project Requirements:

Standard Tools: `numpy` `pandas`

Visualization Tools: `matplotlib` `seaborn` `IPython.display (SVG)` `pydot`

Data Analysis Tools: `scikit-learn` `keras` `tensorflow`

## Project Layout

`utils` contains all the code necessary to run the jupyter notebooks

`models` contains saved models (if model is run and saved)

`logs` contains Tensorboard logs

`figs` contains results and figures for the TRIUMF presentation. 

`data` contains the relevant data needed to run the notebooks. 

All the jupyter notebooks are used for different portions. Done in order:

1. get_contracts - get all the contract information that is already known.

2. format_stats - create usable .csv files for the yearly player stats. 

3. visualize_stats - get a look at some of the interesting stats.

4. training - train the models. 

5. presentation - presentation file for TRIUMF
