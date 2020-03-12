# Data for Assignment 2

Loading the Features and Label Data:

*Python Code (make sure you setup scipy)*

`from scipy.io import loadmat`<br/>
`import numpy as np`<br/>
`X = loadmat(r"train.mat")`<br/> 
`y = np.loadtxt(r"train.targets")`<br/>
`X = X['X'].todense()`<br/>
`print(X.shape)`<br/>

`def LogisticLoss(w, X, y, lam):`<br/>
    `# Computes the cost function for all the training samples`<br/>
    `m = X.shape[0]`<br/>
    `Xw = np.dot(X,w)`<br/>
    `yT = y.reshape(-1,1)`<br/>
    `yXw = np.multiply(yT,Xw)`<br/>
    `f = np.sum(np.logaddexp(0,-yXw)) + 0.5*lam*np.sum(np.multiply(w,w))`<br/>
    `gMul = 1/(1 + np.exp(yXw))`<br/>
    `ymul = -1*np.multiply(yT, gMul)`<br/>
    `g =  np.dot(ymul.reshape(1,-1),X) + lam*w.reshape(1,-1)`<br/>
    `g = g.reshape(-1,1)`<br/>
    `return [f, g]`<br/>
        
