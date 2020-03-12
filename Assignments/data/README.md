# Data for Assignment 2

Loading the Features and Label Data:

*Python Code (make sure you setup scipy)*

`from scipy.io import loadmat`<br/>
`import numpy as np`<br/>
`X = loadmat(r"train.mat")`<br/> 
`y = np.loadtxt(r"train.targets")`<br/>
`X = X['X'].todense()`<br/>
`print(X.shape)`<br/>

`def LogisticLoss(w, X, y, lam):
    # Computes the cost function for all the training samples
    m = X.shape[0]
    Xw = np.dot(X,w)
    yT = y.reshape(-1,1)
    yXw = np.multiply(yT,Xw)
    f = np.sum(np.logaddexp(0,-yXw)) + 0.5*lam*np.sum(np.multiply(w,w))
    gMul = 1/(1 + np.exp(yXw))
    ymul = -1*np.multiply(yT, gMul)
    g =  np.dot(ymul.reshape(1,-1),X) + lam*w.reshape(1,-1)
    g = g.reshape(-1,1)
    return [f, g]
        
