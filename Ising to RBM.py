#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[16]:


# Creating Ising spins and Calculating Hamiltonian of the Ising system
class Ising():
    
    def __init__(self, nRow, nCol):
        self.spins = torch.zeros(nRow, nCol)
        self.probs = torch.rand(nRow, nCol)
        for i in range(nRow):
            for j in range(nCol):
                if self.probs[i][j] < 0.5:
                    self.spins[i][j] = 1
                else:
                    self.spins[i][j] = -1
    
    def Hamiltonian(self):
        H = 0.
        J = 1.
        nRow = self.spins.size()[0]
        nCol = self.spins.size()[1]
        for i in range(nRow):
            for j in range(nCol):
                if i < 1:
                    H -= J * self.spins[i][j] * self.spins[i+1][j]
                elif i > nRow - 2:
                    H -= J * self.spins[i][j] * self.spins[i-1][j]
                else:
                    H -= J * self.spins[i][j] * self.spins[i+1][j]
                    H -= J * self.spins[i][j] * self.spins[i-1][j]
                
                if j < 1:
                    H -= J * self.spins[i][j] * self.spins[i][j+1]
                elif j > nCol - 2:
                    H -= J * self.spins[i][j] * self.spins[i][j-1]
                else:
                    H -= J * self.spins[i][j] * self.spins[i][j+1]
                    H -= J * self.spins[i][j] * self.spins[i][j-1]
        return H/2   #to avoid double count


# In[13]:


# Creating the RBM Architecture (weights, biases)
class RBM():
    
    # Initiate RBM parameters
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(nh)
        self.b = torch.randn(nv)
    
    def Hamiltonian(self, v, h): 
        ah = torch.dot(self.a, h)
        bv = torch.dot(self.b, v)
        hWv = torch.dot(h, torch.mv(self.W, v))
        H = - ah - bv - hWv
        return H
    
    def FreeEnergy(self, v):
        bv = torch.dot(self.b, v)
        Wv = torch.mv(self.W, v)
        F = - bv
        for i in range(Wv.size()[0]):
            F -= torch.log(1 + torch.exp(self.a[i] + Wv[i]))
        return F
    
    # Calculate p(v = D[i]) using Softmax
    def p_v(self, D):
        # Free Energies of each v = D[i]
        F = torch.tensor(D.size()[0])
        for i in range(D.size()[0]):
            F[i] = self.FreeEnergy(D[i])
            
        # p(v = D[i]) = Softmax(-F)[i] = exp(-F[i])/Z
        p_v = F.softmax(- F, dim = 0)
        return p_v
    
    # Calculate Negative Log-Likelihood using log_softmax
    def NLL(self, D):
        # Free Energies of each v = D[i]
        F = torch.zeros(D.size()[0])
        for i in range(D.size()[0]):
            F[i] = self.FreeEnergy(D[i])
            
        # p(v = D[i]) = Softmax(-F)[i] = exp(-F[i])/Z
        LSM = F.log_softmax(- F, dim = 0)
        NLL = - torch.mean(LSM)
        return NLL
    
    def sigmoid_i(self, D, idx):
        a = self.a
        WD_i = torch.mv(self.W, D[idx])
        sigmoid = torch.sigmoid(a + WD_i)
        return sigmoid
    
    def grad_F_i(self, D, idx, param):
        
        grad_F_i = torch.zeros_like(param)
        
        if param == self.W:
            for j in range(grad_F_i.size()[0]):
                for k in range(grad_F_i.zie()[1]):
                    grad_F_i[j,k] = - self.sigmoid_i(D, idx)[j] * D[idx][k]
        
        elif param == self.a:
            for j in range(grad_F_i.size()[0]):
                grad_F_i[j] = - self.sigmoid_i(D, idx)[j]
        
        elif param == self.b:
            for j in range(grad_F_i.size()[0]):
                grad_F_i[j] = - D[idx][j]
        
        return grad_F_i
        
    # Gradients of Negative Log-Likelihood
    def grad_NLL(self, D, param):
        
        grad_NLL = torch.zeros_like(param)
        nData = D.size()[0]
        
        for idx in range(nData):
            grad_NLL += (1 / nData - self.p_v(D)[idx]) * self.grad_F_i(D, idx, param)
        
        return grad_NLL
    
    # Update the RBM parameters
    def update(self, D, learning_rate):

        grad_NLL_w = self.grad_NLL(D, self.W)
        grad_NLL_a = self.grad_NLL(D, self.a)
        grad_NLL_b = self.grad_NLL(D, self.b)
        
        self.W -= learning_rate * grad_NLL_w
        self.a -= learning_rate * grad_NLL_a
        self.b -= learning_rate * grad_NLL_b


# In[5]:


# Fuction to create a training data set by using Metropolis algorithm
def Data_for_train(nData, nRow, nCol):

    for i in range(nData):
    
        ising = Ising(nRow, nCol)
        H_new = ising.Hamiltonian()
    
        # Reshape of a matrix of Ising spins to a vector as the visible layer
        spin = ising.spins.view(nRow*nCol)
        v = (1 - spin)/2   # spin 1 --> 0 ,   spin -1 --> 1
    
        # save visible layers as row vectors of the training data matrix
        if i == 0:
            data = v.unsqueeze(0)
        else:
            if H_new <= H:
                data = torch.cat((data, v.unsqueeze(0)), dim = 0)
            else:
                B_dist = dist.Bernoulli(torch.exp(H - H_new))
                B = B_dist.sample()
                if B == 1:
                    data = torch.cat((data, v.unsqueeze(0)), dim = 0)
                else:
                    data = torch.cat((data, data[i-1].unsqueeze(0)), dim = 0)
        H = H_new
        
    return data


# initiate the data set and RBM parameters

# In[6]:


# make a training data set
D = Data_for_train(nData = 10, nRow = 3, nCol = 3)
print(D)


# In[11]:


# initiate RBM parameters
rbm = RBM(nv = D.size()[1], nh = 4)


# Training

# In[15]:


# Train the RBM
num_epoch = 100
#batch_size = 1\
lr = 1e-3  #learning_rate

for epoch in range(0, num_epoch + 1):
        
    if epoch > 0:
        rbm.update(D, lr)
    
    print('epoch {}: W = {}'.format(epoch, rbm.W))
    print('\t a = {}'.format(rbm.a))
    print('\t b = {}'.format(rbm.b))
    #print('\t loss = {}'.format(rbm.NLL(D)))    


# In[ ]:




