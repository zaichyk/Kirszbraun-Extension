#!/usr/bin/env python
# coding: utf-8

# This notebook provides an implemetation of the MWU Kirszbraun algrorithm as discribed in our paper [] (TBD).
# the algorthim evalutes a non parametric function $f:X \rightarrow Y$ by a consistent lischitz function. for more details please read []. The algorithm works in two phases:
# 1. Smoothing - where we evaluate the training points, based on seeing data X_train,Y_train, so they fit a smooth lipschitz funtion. The Lischitz constant is picked using a cross validation search on several candidates.
# 2. Extension. Based on the smoothing, each new point can be evaluated in a way that the lipschitz constant is reserved. 
# 
# you can run this code on either "cpu" or "gpu".
# 
# 
# We start by import crucial packages, make sure you have them installed.

# In[1]:


import torch
import time
import pandas as pd
import numpy as np
from math import ceil,log,pi,inf,cos,sin,sqrt
from sklearn.metrics.pairwise import euclidean_distances
from IPython.core.debugger import Tracer;


# ## 1. Smoothing##
# Smoothing finds an evaluation of the X_train data, such that keep a fixed Lipschitz constant, that minimze the overall loss of the observed data Y_train.  $\Phi(\tilde{Y},Y) = \frac{1}{n}\sum_{i=1}^n ||y_i-\tilde{y_i}||$.
# fixing the Lipshitz constant, allow us to prevent over fitting.
# The main algorithm is in SmoothMWU. FindSolutionAndReport and Smooth functions are wrapping of the algorithms.

# In[2]:


gpu_num = 0
device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")
if torch.cuda.is_available():
    print(f"{torch.cuda.get_device_name(gpu_num)}")

eps = 0.1 #epsilon - the precision


# the running time can improved by a factor of 2
# since we don't have to compute both |y_i - y_j| and |y_j - y_i|
def PairwiseSqDistances(Y):
    #Y is is an N-by-dimB matrix
    [N, dimB] = list(Y.size())

    repY  = Y.unsqueeze(1).expand(N, N, dimB)
    repYt = repY.transpose(0, 1)
    return torch.sum((repY - repYt) ** 2, 2)
    #return torch.Tensor(euclidean_distances(Y)**2)
    
    
def findLipschitzConstant(X,Y):
    """
    given two sets of vectors of size n where f(X_i) = Y_i
    finds the Lipschitz constant of f over (X,Y). 
    """
    
    #return (torch.max(PairwiseSqDistances(Y) / (PairwiseSqDistances(X)+torch.eye(X.size()[0],device = device)))).to(device)
    distY = torch.Tensor(euclidean_distances(Y,Y)**2)
    distX = torch.Tensor(euclidean_distances(X,X)**2) 
    return  (torch.max(distY / (distX+torch.eye(distX.size()[0],device = device)))).to(device)
    
def SmoothMWU(Y, R_sq, Phi_0):
    # Y is an N-by-dimB matrix; Y^T from the write-up
    # R_sq is is an N-by-N matrix    
#    from IPython.core.debugger import Tracer; Tracer()()
    [N, dimB] = list(Y.size())
    
    m = N*(N-1) // 2 # the number of pair of points,
    delta = eps / (2*m) #what is delta

    # Initialize the weight matrix    
    w_Phi = 0.5
    W = torch.Tensor(N,N).to(device).fill_(0.5 / m)
    for i in range(N):
        R_sq[i,i] = 1 # why?

    T = 350 # used for speed, for better resalts use sqrt(m)*log(N)/eps**2)

    YSmoothed = torch.zeros(N, dimB, device = device) # returned output. initial guess

    for t in range(T):
        lambda_inv = Phi_0 / (w_Phi + eps/2) ## \lambda^{-1}           

        # set off-diagonal entries

        L = -lambda_inv * torch.div(W + delta, R_sq)
        
        # set/fix the diagonal entries
        S = torch.sum(L, dim = 1)
        for i in range(N):
            L[i,i] += -S[i] + 1

        # solve for Yt ; gesv ==> solve in the next version of torch
        Yt, _ = torch.gesv(Y, L)  
        YSmoothed *= t/(t+1)
        YSmoothed += Yt/(t+1)

        # update the weights
        # first update W, that is w_{ij}'s 
        PD = PairwiseSqDistances(Yt)
        WUpdate = 1 + 2*eps * (torch.div(PD, R_sq) ** 0.5 - 1)
        W *= WUpdate

        # now update w_Phi
        WPhiUpdate = 1 + 2*eps * ((torch.sum((Y - Yt) ** 2) / Phi_0) ** 0.5  - 1) 
        w_Phi *= WPhiUpdate        
               
        # renormalize
        for i in range(N):
            W[i,i] = 0

        TotalW = torch.sum(W) / 2 + w_Phi
        W /= TotalW
        w_Phi /= TotalW

    return YSmoothed
        
def Smooth(Y, R_sq):
    [N, dimB] = list(Y.size())
    YMean = (torch.sum(Y, dim = 1) / N).unsqueeze(1).expand(N, dimB)

    Phi_0 = torch.sum((Y - YMean) ** 2) #a very crude upper bound
    
    PhiUB = Phi_0
    PhiLB = 0
    
    while (PhiUB > (1 + eps/10) * PhiLB):
        print("<", end = "")
        Phi_0 = (PhiLB + PhiUB) / 2
        YSmoothed = SmoothMWU(Y, R_sq, Phi_0)
        bLip = ( PairwiseSqDistances(YSmoothed) < (1 + eps) * R_sq ).all()
        #bLip = ( PairwiseSqDistances(YSmoothed) < (1 + eps) * R_sq ).all()
        bPhi = (torch.sum((Y - YSmoothed) ** 2) / Phi_0) < 1 + 2 * eps

        if bLip and bPhi:
            PhiUB = Phi_0
        else:
            PhiLB = Phi_0   
        print(">", end = "", flush = True)         
 
    print()
    return SmoothMWU(Y, R_sq, PhiUB)

def FindSolutionAndReport(Y, X,Lip, bReportVectors=True):
    R_sq = PairwiseSqDistances(X)
    #R_sq = torch.Tensor(euclidean_distances(X,X))
    StartTime = time.time()
    YSmoothed = Smooth(Y,Lip * R_sq)
    phi = torch.mean((Y - YSmoothed) ** 2).item()
    #d,n = X.size()[1]+Y.size()[1],X.size()[0] # important for risk bound using Rademacher complexity
    #k = ( (d-1)*34*(4*Lip)**(d/2) ) / ( 2*sqrt(n) )
    #rad = 8*k**(1/(d+1)) + d*k**((d-1)/(d+1)) - 2*(-(d+1)/2)
    #rad_bound = phi+rad
    print("Phi: ", phi)#, " Rad:", rad, " Bound: ",rad_bound)
    print("Lipschitz constant ",findLipschitzConstant(X,YSmoothed))
    if bReportVectors:
        print("vectors:", YSmoothed)

    print("Elapsed time", round(time.time()- StartTime,2), " sec")
    return YSmoothed #, rad_bound


# ## 2. Extension ##
# The extension evaluate a new point $x \in X$ that preserved the lipschitz constant of the Z_train set 

# In[3]:


def nearestNeighbor(x,X,Y):
#NEARESTNEIGHBOR finds the nearest neighbor and it's distance
#   x - a vector
#   X - group of vectors represented as row vectors.
#   returns the nearest neigbor of x in X, the ditance ||x-nn||, and
#   y0=f(x0)

    N = X.size()[0]; # number of vectors in X
    difference = torch.norm(X-x.repeat(N,1),dim=1)
    dist,i = torch.min(difference),torch.argmin(difference)
    nn = X[i,:]
    y0 = Y[i,:]
    return dist,y0

def extension(x,X,Y,eps,L):
    # extension finds a (1+eps) Lipshitz approximation for z=h(x)
    #   x = the query point (vector)
    #   X = the training set samples
    #   Y = the training set labeling (not the true lables). actualy Y=Z
    #   eps = the approximation parameter. in (0,1/2)
    #   z = the output , the extension of h to x. z=h(x) s.t the lipshitz
    #   constnant is less then (1+eps)L
    ## the marks numbers are for being consistent with the paper of the work.
    
    # basic parameters
    n = X.size()[0]; # sample size
    DOF2 = Y.size()[1] # dimension of Y 
    
    #1. find nearest neighbour of x out of X; y0 = f(x0) ; d0 = ||x0-x||
    d0,y0 = nearestNeighbor(x,X,Y)
    #2. T is nuber of iterations
    T = 1000 # for better results use torch.min(ceil(16*log(n)/eps**2)
    #3. initialize weights vector w where w1=1/n for each i
    w = torch.ones(n,T+1).to(device)*1/n;
    #4. initialize distance vector d where di = ||xi - x||
    d = L*torch.norm(X-x.repeat(n,1),dim=1)
    #5. Steps 6-10 will be repeated until convergence
    for t in range(T):
        #6. create a distribution
        P = torch.sum(w[:,t]/(d**2)); #normaplization parameter
        p = w[:,t]/(P*(d**2));
        #7. z0 = sum(pi*yi) and delta = ||z0-y0||
        z0 = torch.sum(Y*p.repeat(DOF2,1).transpose(0,1),dim=0)
        delta = torch.norm(z0-y0);
        #8. evalute z
        if delta <= d0:
            z = z0
        else:
            z = d0/delta*z0+(delta-d0)/delta*y0
        #9. update weights : wi(t+1) = (1+(eps*||z-yi||/8di))*wi(t) for all i 
        tmp_dist = torch.norm(z.repeat(n,1)-Y,dim=1)
        w[:,t+1]=torch.ones(n,device=device)+eps*tmp_dist/(8*d)*w[:,t]        
        #10. normalize the weitghts
        W = torch.sum(w[:,t+1]);
        w[:,t+1] = (1/W)*w[:,t+1];
    #11. average over weights
    final_w = torch.sum(w,dim=1)*1/(T+1);
    #12. compute z as in 6-8
    #6. 
    P = torch.sum(final_w*(d**2)); #normaplization parameter
    p = final_w/(P*(d**2));
    #7. z0 = sum(pi*yi) and delta = ||z0-y0||
    z0 = torch.sum(Y*p.repeat(DOF2,1).transpose(0,1),dim=0)
    delta = torch.norm(z0-y0);
    #8. evalute z
    if delta <= d0:
        z = z0
    else:
        z = d0/delta*z0+(delta-d0)/delta*y0
    return z

def Test(X_val,X_train,Z_train,eps,l):
    n_val = X_val.size()[0] # validation/test set size
    DOF2 = Z_train.size()[1];    # dimension of Y, the second agent
    Z_val = torch.ones(n_val,DOF2).to(device)
    for i in range(n_val):
        x0 = X_val[i]
        Z_val[i] = extension(x0,X_train,Z_train,eps,l)
#debug        print(Z_val[i])
    return Z_val


# ## 3. Train and Test
# the next section is for using the algorithm in order to train and test the algorithm.
# The training consist of finding the optimal Lipshchitz $l$ constant using cross validation over different $l$ candidates.
# We also used the Structur Risk Minimization (SRM) which finds the $l$ which yields to the smallest generalization bound, but in practice this factor is non informative unless n is on a very large scale (milions). 
# It is likely the SRM will be deleted in future for now you can uncomment it for your own use

# In[1]:


def lstsqrs(Z,Y):
    ### will return the distance phi = sum(||Z-Y||**2)
    ### when dealing with angels we notice that distance(0,2pi) = 0
    N = Z.size()[0]
    return torch.sum(torch.norm(Z-Y,dim=1)**2)/N


#mod_lstqr consider the periodic of distance between angles
def mod_lstsqr(Z,Y):
    ### will return the distance phi = sum(||Z-Y||**2)
    ### when dealing with angels we notice that distance(0,2pi) = 0
    diff = torch.min((Z-Y)%(2*pi),(Y-Z)%(2*pi))
    dist = torch.norm(diff,dim=1)**2
    return dist.mean()
    
def crossVal(X,Y,X_val,Y_val,k_fold = 10):
    #basic parameters
    N,DOF1 = X.size();
    DOF2 = Y.size()[1];
    N_val = X_val.size()[0]
    lip = findLipschitzConstant(X,Y)
    L = torch.exp(torch.linspace(0,torch.log(lip),k_fold))
    StartTime = time.time()
    
    #important values
    
    no_improvement  = 0
    lip_const  = inf #the returned lipschitz constant of the CV process
    Phi = inf # the returned score score sum(||Y_i-Z_i||)/N , 
    Z_train = torch.zeros(Y.size()), #the returned smoothing
    
    for l in L:
        tmp_time = time.time()
        Z_train_l = FindSolutionAndReport(Y,X,l,False) # smoothig by l        
        Z_val = Test(X_val,X,Z_train,eps,l) # validating
        Phi_l = lstsqrs(Z_val,Y_val) 
        print("lip ",l," Phi = ",Phi_l, "time = ", round(time.time()- tmp_time,2), "sec")
        
        if Phi_l < Phi:
            Phi = Phi_l
            lip_const = l
            Z_train = Z_train_l
            no_improvement = 0
            
        else:
            no_improvement += 1
            
        if no_improvement >= 2: # if it is the second time it means we start over fitting so we stop
            print("lip ",l," Phi = ",Phi_l, "time = ", round(time.time()- tmp_time,2), "sec")
            print("We start overfitting the data")
            break
        
        
    print("finished training ",N," points ",k_fold,"crossValidation in ",round(time.time()- StartTime,2)," sec")
    print("Lip_const_cv: l=", lip_const," with score: Phi=",Phi)
    
    return Z_train,lip_const


# For Testing our model, we first import the data, which should be allocated in a directory named data - placed in the same directory as this code file

# In[5]:


# if you want to learn py positioning (may lead to distortions)
N_train = 10000
N_test = 50
data_dir = 'data_3to5_positions'
X = torch.Tensor(np.array(pd.read_csv(data_dir+'\posX_train.csv', header = None))[:N_train,:]).to(device)
Y = torch.Tensor(np.array(pd.read_csv(data_dir+'\posY_train.csv', header = None))[:N_train,:]).to(device)
#noise = torch.randn(Y.size()) * Y.std()/10
#Y = Y + noise
X_test = torch.Tensor(np.array(pd.read_csv(data_dir+'\posX_test.csv', header = None))[:N_test,:]).to(device)
Y_test = torch.Tensor(np.array(pd.read_csv(data_dir+'\posY_test.csv', header = None))[:N_test,:]).to(device)


# In[6]:


#TODO: this temporary box should be integrated in the cross validation function
from random import sample
train_size = N_train*9//10
R = sample(range(N_train),train_size+train_size//10)
I,J = R[:train_size],R[train_size:]
X_train,X_val = X[I],X[J]
Y_train,Y_val = Y[I],Y[J]


# For thousands of points running time of FindAndReport could be in hours. therefor we find the Lipschitz constant via Cross valditaion over only part of the training set. Once we picked the lip_const of the problem, we can train smooth the entire training set.

# In[ ]:


#CV_N = 1000
Z_train,lip_const = crossVal(X_train,Y_train,X_val,Y_val,k_fold=10)


# In[10]:


Z_train =  FindSolutionAndReport(Y_train,X_train,lip_const,False)


# Use the following code if you already run the algorithm and\or want to save time for the exploration part

# In[2]:


#Z_train = torch.Tensor(np.array(pd.read_csv(data_dir+'\Z_train.csv', header = None))[:N_train,:]).to(device)
#Z_test = torch.Tensor(np.array(pd.read_csv(data_dir+'\Z_test_mw.csv', header = None))[:N_test,:]).to(device)
#lip_const = findLipschitzConstant(X_train,Z_train)


# In[13]:


startTime = time.time()
Z_test = Test(X_test,X_train,Z_train,eps,lip_const)
print("Test ",Z_test.size()[0],"samples in ",  round(time.time()- startTime,2)," sec.")


# Let's see how we did.

# In[9]:


print("AVG lstsqrs: ",lstsqrs(Z_test,Y_test)/N_test)


# ## 4 . comparison
# To see how well the algorithm performed we compare it with a multi SVR competitor using a Gausian kernel.
# The competitor is a simply a multiple regressors $g_i:X\rightarrow Y_i$, where $Y_i$ is the i coordinate of Y.
# Meaning, let $x\in X$ and k be the dimension of $y \in Y$ $g(x) = (g_1(x),...,g_k(x))$.
# Unlike the MWU implementation, this section is not modular in the dimensionality of $Y$, thus adding more regressors might be needed

# ## Nadaraya Watson

# In[10]:


def lstsqrs(Y_hat,Y):
    ### will return the distance phi = sum(||Z-Y||**2)
    ### when dealing with angels we notice that distance(0,2pi) = 0
    N = Y_hat.shape[0]
    return np.sum(np.linalg.norm(Y_hat-Y,axis=1)**2)/N

def K0(X,x,h):
    #gets a vector and returns 1 if the condition is held
    ind = np.linalg.norm((X - x),axis=1)/h < 1 # indicator
    return ind.astype(int)

def Rec_Kernel(X,x,h):
    1/2*K0(X,x,h)

    
def gau_Kernel(X,x,h, sigma2 = 1):
    dist = np.linalg.norm((X-x),axis=1)**2
    exp = np.exp(dist/sigma2)
    return exp

def f_NW(X,Y,x,Kernel,h=1):
    #X and Y are the training set and labels, x a new point,K is a Kernel, h is the bandwith
    K = Kernel(X,x,h)
    nominator = np.sum( np.transpose(Y.transpose() * K ),axis = 0)
    deliminator = sum(K)
    return (0 if deliminator== 0 else nominator/deliminator)

def predict(X_tr,Y_tr,X_test,Kernel = K0, h = 1):
    n,d = X_test.shape[0],Y_tr.shape[1]
    Y_hat = np.zeros([n,d])
    for i in range(n):
        x = X_test[i]
        Y_hat[i] = f_NW(X_tr,Y_tr,x,Kernel)
    return Y_hat

def fit(X,Y,Kernel = K0 , k_fold = 20):
    #retruns the optimal bandwith h
    if k_fold < 1: k_fold = 1
    n = X.shape[0] #sample size
    H = np.linspace(0.0001,1,k_fold)
    j = 0
    lst_sqr = np.zeros(k_fold)
    timer = time.time()
    print("Strat fitting the data")
    for h in H:
        start = time.time()
        print("fitting h=",h)
        lst_l = 0
        for i in range(2): # we will do 10 random ordering validation on each h for better consistency with the results
            shuffle = np.random.permutation(range(n))
            N_train = n*9//10
            X_train,X_val = X[shuffle[:N_train]],X[shuffle[N_train:]]
            Y_train,Y_val = Y[shuffle[:N_train]],Y[shuffle[N_train:]]
            Y_hat = predict(X_train,Y_train,X_val,Kernel,h)
            lst_sqr[j] = lst_sqr[j] + lstsqrs(Y_hat,Y_val) 
        lst_sqr[j] = lst_sqr[j]/10
        end = time.time()
        print("finished with score: ", lst_sqr[j]," time: ",(end - start))
        j = j+1
    h_opt = H[np.argmin(lst_sqr)]
    end = time.time()
    print("Finished fitting in: ", (end - timer)," Optimal h: ", h_opt)
    return h_opt


# In[41]:


h = fit(X.numpy(),Y.numpy())


# In[42]:


Y_hat_test = predict(X.numpy(),Y.numpy(),X_test.numpy(),h=h)
lstsqrs(Y_hat_test,Y_test.numpy())


# ## plot

# In[12]:


import matplotlib.pyplot as plt


# In[82]:


def plot4Arms(arm1,arm2,arm3,arm4):
    x1,y1 = arm1[0::2],arm1[1::2] #expert
    x2,y2 = arm2[0::2],arm2[1::2]
    x3,y3 = arm3[0::2],arm3[1::2]
    x4,y4 = arm4[0::2],arm4[1::2]
    plt.plot(x1,y1,linestyle = '-',marker='o',color = 'k') # expert (k link)
    plt.plot(x2,y2,linestyle = '--',marker='o',color = 'r') # true (d link)
    plt.plot(x3,y3,linestyle = '-',marker='o',color = 'y') # competetor (d link)
    plt.plot(x4,y4,linestyle = '-',marker='o',color = 'b') # MWU (d link)

def plot3Arms(arm1,arm2,arm3):
    x1,y1 = arm1[0::2],arm1[1::2]
    x2,y2 = arm2[0::2],arm2[1::2]
    x3,y3 = arm3[0::2],arm3[1::2]
    plt.plot(x1,y1,linestyle = '--',marker='o',color = 'r') #true
    plt.plot(x2,y2,linestyle = '-',marker='o',color = 'y') #competetor
    plt.plot(x3,y3,linestyle = '-',marker='o',color = 'b') #MWU

def plot2Arms(arm1,arm2):
    x1,y1 = arm1[0::2],arm1[1::2]
    x2,y2 = arm2[0::2],arm2[1::2]
    plt.plot(x1,y1,linestyle = '--',marker='o',color = 'r') #true
    plt.plot(x2,y2,linestyle = '-',marker='o',color = 'k') #learner


# We can examine the MW learner final positions (where it matters), ploting the learned correspondence.
# We will plot 2 arms on the same figure:
# 1. black = the expert arm (5 degrees of freedom)
# 2. Red\Dashed = y. The true, *unknown* correspondence of the learner. It can easily spotted the the end frame of the learner and expert should always coinside. 3 degrees of freedom
# 3. Blue = z_mw The learned correspondence by the MWU learner.
# 4. Yellow = the competetor - Nadaraya Watson kernel regression.

# In[83]:


I = sample(range(N_test),20); # change for as much figures you want to plot
for i in I:
    print(i,":")
    plot2Arms(Y_test[i].numpy(),Z_test[i].numpy())
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    plt.axis('equal')
    plt.show()
    plt.clf()


# We can examine the MW learner final positions vs the NW, ploting the learned correspondence by each algorithm\. 
# We will plot 3 arms on the same figure:
# 
# Black = y. The true, unknown correspondence of the learner. It can easily spotted the the end frame of the learner and expert should always coinside. 3 degrees of freedom
# Blue = z_mw The learned correspondence by the MWU learner
# Yellos = z_nw The learned correspondence by the Nadaraya-Watason kernel

# In[84]:


I = [5,15,19,28]# change for as much figures you want to plot
for i in I:
    plt.clf()    
    print(i,":")
    plot3Arms(Y_test[i].numpy() , Y_hat_test[i], Z_test[i].numpy())
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    plt.axis('equal')
    plt.show()
    plt.clf()


# In[80]:


I = range(5) # change for as much figures you want to plot
for i in I:
    plt.clf()    
    print(i,":")
    plot2Arms(Y_test[i].numpy() , X_test[i].numpy())
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    plt.axis('equal')
    plt.show()
    plt.clf()


# In[ ]:




