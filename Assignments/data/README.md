# Data for Assignment 2

Reading the Features and Label File:

*Python Code (make sure you setup scipy)*

`from scipy.io import loadmat`<br/>
`import numpy as np`<br/>
`X = loadmat(r"train.mat")`<br/> 
`y = np.loadtxt(r"train.targets")`<br/>
`X = X['X'].todense()`<br/>
`print(X.shape)`<br/>
