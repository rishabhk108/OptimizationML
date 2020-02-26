# Data for Assignment 2

Reading the Features and Label File:

*Python Code (make sure you setup scipy)*

`from scipy.io import loadmat`.   
`import numpy as np`. 
`X = loadmat(r"train.mat")`. 
`y = np.loadtxt(r"train.targets")`. 
`X = X['X'].todense()`. 
`print(X.shape)`. 
