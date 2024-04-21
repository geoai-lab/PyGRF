# PyGRF: An improved Python Geographical Random Forest model and case studies in public health and natural disasters

### Overall description
Geographical random forest (GRF) is a recently developed and spatially explicit machine learning model. With the ability to provide more accurate predictions and local interpretations, GRF has already been used in many studies. Implemented as an R package, GRF currently does not have a Python version which limits its adoption among machine learning practitioners. The current GRF model also has limitations in the determination of the local model weight and bandwidth parameters, potential insufficient numbers of local training samples, and sometimes high local prediction errors. In this work, we develop a Python-based GRF model, PyGRF, and conduct assessment to ensure its consistency with the original R-based model. We further address the three limitations by introducing theory-informed local model weight and bandwidth determination, local training sample expansion, and spatially-weighted local predictions. We demonstrate the performance of PyGRF and use it in two case studies in public health and natural disasters.

This repository contains the source code and parameter descriptions of the PyGRF model, and two Jupyter Notebooks and related datasets for the two case studies.


<br />
<br />

<p align="center">
<img align="center" src="Figs/Consistency.jpg" width="600" />
<br />
Figure 1. Scatter plots for predictions of PyGRF and GRF in three different settings: (a) 50 trees; (b) 75 trees; (c) 100 trees.
</p>

<br />
<br />
<p align="center">
<img align="center" src="Figs/Obesity.jpg" width="600" />
<br />
Figure 2. Map visualizations for the case study of obesity prevalence estimation in NYC: (a) city boundary of NYC and its census tracts; (b) obesity prevalence in NYC in 2018.
</p>
<br />

<br />
<p align="center">
<img align="center" src="Figs/311.jpg" width="600" />
<br />
Figure 3. Map visualizations for the case study of help request prediction in Buffalo: (a) city boundary of Buffalo and its census block groups; (b) help requests in Buffalo during the period from 12/19/2022 to 1/1/2023.
</p>
<br />



### Repository organization

* The file "PyGRF.py" is the source code of this Python-based GRF model.
* The file "Description_Parameters.pdf" explains the details of parameters in this model.
* The folder "Notebooks" contains two Jupyter Notebooks used for implementing the two case studies in public health and natural disasters.
* The folder "Data" contains the experimental data for the two case studies including obesity rate data and 311 help request data.
<br />



### Installation

We have published the code for PyGRF as a Python package in [PyPI](https://pypi.org/project/PyGRF/). You can directly install it with the command "pip install PyGRF". 
<br />
<br />



### Example:

Below shows an example on how to fit a PyGRF model and use it to make predictions.

```python
from PyGRF import PyGRF
from sklearn.model_selection import train_test_split

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Create a PyGRF model by specifying hyperparameters
pygrf_example = PyGRF.PyGRFBuilder(n_estimators=60, max_features=1, band_width=39, train_weighted=True, predict_weighted=True, bootstrap=False,
                          resampled=True, random_seed=42)

#Fit the created PyGRF model based on training data and their spatial coordinates						  
pygrf_example.fit(X_train, y_train, xy_coord)

#Make predictions for testing data using the fitted PyGRF model and you specified local model weight 
predict_combined, predict_global, predict_local = pygrf_example.predict(X_test, coords_test, local_weight=0.46)
```


### Parameters:
If you want to learn more about the major parameters in this package, please refer to the [Description of Parameters](https://github.com/geoai-lab/PyGRF/blob/master/Description_Parameters.pdf).



## Authors
* **Kai Sun** - *GeoAI Lab* - Email: ksun4@buffalo.edu
* **Yingjie Hu** - *GeoAI Lab* - Email: yhu42@buffalo.edu


### Reference
If you use the data or code from this repository, or the PyGRF package, we will really appreciate if you can cite our paper:

Kai Sun, Ryan Zhenqi Zhou, Jiyeon Kim, and Yingjie Hu. 2024. PyGRF: An improved Python Geographical Random Forest model and case studies in public health and natural disasters.


