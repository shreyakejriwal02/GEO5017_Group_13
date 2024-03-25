## Name 
Assignment: GEO5017 A2: Group 13

## Description 
This script extracts features from point clouds and performs classification using different machine learning algorithms.  

## Dependencies 
Following packages are required to run this code: 

1. 'os': for file path management.
2. 'numpy': for mathematical calculation and arrays.
3. 'pandas': for reading and managing files.
4. 'matplotlib': for data visualisation.
5. 'sklearn': for machine learning models and performance metrics.
6. 'seaborn': for data visualisation (graphs and heatmaps).

Rhino and grasshopper was used for visualisation of the urban objects to observe and decide the feature set.

## Installation 
Multiple internal and external libraries used like numpy, matplotlib and math 

```bash 
pip install numpy 
pip install pandas
pip install matplotlib 
pip install scikit-learn
pip install seaborn
pip install scipy
```

## Usage 
To use each library and function they must be first imported. 
  
```python 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, learning_curve
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.stats import norm
from scipy.stats import gaussian_kde
``` 
## How to run the script

Loading data: 
The data file needs to be added to the current directory with the name 'pointcloud-500' in order to run the script. The folder contains all the files in .xyz format.

Visualising objects:
To visualize an object in 3D space, you can set the t variable to the index of the object you want to visualize in the point_cloud list. 
When prompted enter a number between 0-499 to see a visualisation of any object.

Run the script
Run the script to extract features and perform classification, get accuracy score and plot the learning curve.

Extracted features:
    1. Max height of the object
    2. Ground area covered by the object through a bounding box
    3. Density of points in reference to bounding box volume
    4. Projected area ratio of object with the bounding box area
    5. Average clustering height of points in the object 
    6. Planarity of the objec

Algorithms used:
    1. Support vector machine
    2. Random forest

Accuracy tested via:
    1. Overall accuracy
    2. Mean per-class accuracy
    3. Confusion matrix

## Output
The script outputs 
1. A visualisation of the object specified by the user in 3D space
2. Feature parameters of all the objects
3. Feature importance table 
4. Confusion matrix and graph
5. Classification report 
6. Accuracy analysis
7. Learning curve of the classification
for each machine learning algorithm used.


## Conclusion
This script provides a framework for extracting features from point cloud data and using those features to perform object classification using machine learning models.

## Credit and references:
All point clouds are taken from the DALES Objects dataset. More details about the dataset can be found in the following paper:
Singer et al. DALES Objects: A Large Scale Benchmark Dataset for Instance Segmentation in Aerial Lidar, 2021.