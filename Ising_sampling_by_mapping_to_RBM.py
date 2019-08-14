#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import itertools
import numpy as np


# In[2]:


def decimal_to_binary_tensor(value, width=0):
    string = format(value, '0{}b'.format(width))
    binary = [0 if c == '0' else 1 for c in string]
    return torch.tensor(binary, dtype=torch.float)


# In[4]:


class RBM():
    
    # Initiate the sizes of visible and hidden layers, and RBM parameters(weights, biases)
    def __init__(self):
        self.num_v = num_rows * num_cols
        self.num_h = 2 * num_rows * num_cols - num_rows - num_cols
        self.W = torch.zeros(self.num_h, self.num_v)
        self.a = torch.zeros(self.num_v)
        self.b = torch.zeros(self.num_h)
        self.model = Ising_model(num_rows, num_cols)
    
    def label_edges(self, i, j):
        num_horizontal_edges = (num_cols - 1) * num_rows
        if i - j == num_cols:
            e = num_horizontal_edges + j
        elif i - j == - num_cols:
            e = num_horizontal_edges + i
        elif i - j == 1:
            e = (num_cols - 1) * int(i / num_cols) + i % num_cols - 1
        elif i - j == -1:
            e = (num_cols - 1) * int(i / num_cols) + i % num_cols
        return e
    
    def get_weights(self, visible):
        J = self.model.J
        for i in range(self.num_v):
            for j in range(i, self.num_v):
                if J[i, j] == 1:
                    e = label_edges(i, j)
                    self.W[e, i] = torch.log(torch.exp(4 * beta * visible[i] * visible[j]) - 1) / (visible[i] + visible[j])
    
    def get_biases(self):
        J = self.model.J
        for i in range(self.num_v):
            self.a[i] = # -2 * beta * (num of 1 in i-th row of J)
        
    def to_hidden(self, visible):
        b = self.b
        Wv = torch.matmul(self.W, visible)
        return torch.sigmoid(b + Wv)

    def to_visible(self, hidden):
        a = self.a
        hW = torch.matmul(hidden, self.W)
        return torch.sigmoid(a + hW)


# In[5]:


class Ising_model():
    
    # initiate the size of 2d lattice, the (inverse-)temperature, interactions
    def __init__(self, num_rows, num_cols, beta):
        self.num_rows, self.num_cols = num_rows, num_cols
        self.num_spins = num_rows * num_cols
        self.beta = beta
        self.J = self.interactions()
    
    # find nearest neighbors in 2d lattice
    def neighbors(self, i, j):
        nhb = []
        if i > 0:
            nhb.append([i-1, j])
        if i < self.num_rows - 1:
            nhb.append([i+1, j])
        if j > 0:
            nhb.append([i, j-1])
        if j < self.num_cols - 1:
            nhb.append([i, j+1])
        return nhb
    
    # ferromagnetic interactions only with nearest neighbors
    def interactions(self):
        J = torch.zeros(self.num_spins, self.num_spins)
        for i, j in itertools.product(range(self.num_rows), range(self.num_cols)):
            for i_nhb, j_nhb in self.neighbors(i, j):
                J[self.num_cols * i + j, self.num_cols * i_nhb + j_nhb] = 1                
        return J
    
    # calculate Hamiltonian of Ising model for a given spin state
    def hamiltonian(self, spin_state):
        s = spin_state
        J = self.J
        H = - torch.chain_matmul(s, J, s) / 2
        return H
    
    def magnetisation(self, spin_state):
        m = torch.mean(spin_state)
        return m
    
    def partition_function(self):
        num_state = 2**(self.num_spins)
        Z = 0.
        for i_state in range(num_state):
            binary = decimal_to_binary_tensor(i_state, width=self.num_spins)
            spin_state = 1 - 2 * binary
            H = self.hamiltonian(spin_state)
            Z += torch.exp(- self.beta * H)
        return Z
    
    # calculate the expectation value of the Energy of Ising model with Boltzmann distribution
    def energy_expectation(self):
        num_state = 2**(self.num_spins)
        Z = self.partition_function()
        E_true = 0.
        for i_state in range(num_state):
            binary = decimal_to_binary_tensor(i_state, width=self.num_spins)
            spin_state = 1 - 2 * binary
            H = self.hamiltonian(spin_state)
            E_true += H * torch.exp(- self.beta * H)
        return E_true / Z
    
    def entropy(self):
        Z = self.partition_function()
        return self.beta * self.energy_expectation() + torch.log(Z)


# In[ ]:


class Ising_sampler():
    
    # initiate the size of a dataset, and create a sampled dataset
    def __init__(self, data_size, num_rows, num_cols):        
        self.model = Ising_model(num_rows, num_cols)
        self.data_size = data_size
        self.data_length = num_rows * num_cols
        self.num_rows, self.num_cols = num_rows, num_cols
        self.dataset = self.gibbs_sampling()
        self.rbm = RBM()
    
    # randomly generate one sample of an spin state as 2d tensor
    def gen_sample(self, p = 0.5):
        probs = torch.rand(self.num_rows, self.num_cols)
        sample = torch.where(probs < p, torch.zeros(1), torch.ones(1))
        return sample
    
    # create a training data set by using Metropolis algorithm
    def gibbs_sampling(self):
        beta = self.model.beta
        num_burn = 10000
        # randomly sample a ising model
        sample_2d = self.gen_sample()
        # reshape a 2d sample to a 1d tensor as the visible layer
        v_current = sample_2d.view(-1)
        # save the sample of the visible layer to the dataset
        dataset = v_current.unsqueeze(0)
        
        for _ in range(num_burn + self.data_size - 1):
            # calculate RBM parameters W and a from a given visible layer by Ising-RBM mapping
            v_current = v_next
            rbm.W = rbm.get_weights(v_current)
            rbm.a = rbm.get_biases()
            pr_h = rbm.to_hidden(v_current)
            h_next = torch.bernoulli(pr_h_next)
            pr_v = rbm.to_visible(h_next)
            v_next = torch.bernoulli(pr_v)
            dataset = torch.cat((dataset, v_next.unsqueeze(0)), dim = 0)
        
        dataset_after_burning = dataset[num_burn : num_burn + self.data_size]
        return dataset_after_burning


# In[ ]:


class Analyze_data():
    
    def __init__(self, dataset):
        self.data_size = dataset.size()[0]
        self.data_length = dataset.size()[1]
        self.dataset = dataset
    
    # calculate the probability of each samples in the dataset by counting
    def prob_data(self):
        num_state = 2**self.data_length
        count = torch.zeros(num_state)
        for i_state in range(num_state):
            for i_data in range(self.data_size):
                bin_state = decimal_to_binary_tensor(i_state, width=self.data_length)
                if torch.all(torch.eq(self.dataset[i_data], bin_state)) == 1:
                    count[i_state] += 1
        prob = count / self.data_size
        return prob
    
    # calculate the Entropy of the dataset
    def entropy_data(self):
        num_state = 2**self.data_length
        prob = self.prob_data()
        entropy = 0.
        for i_state in range(num_state):
            if prob[i_state] > 0:
                entropy -= prob[i_state] * torch.log(prob[i_state])  
        return entropy
    
    # calculate the mean value of the Energy of each samples' in the dataset
    def energy_data(self, model):
        spinset = 1 - 2 * self.dataset
        energy = torch.zeros(self.data_size)
        for i_data in range(self.data_size):
            energy[i_data] = model.hamiltonian(spinset[i_data]).item()
        energy_data = torch.mean(energy)
        return energy_data
    
    def magnetisation_data(self, model):
        spinset = 1 - 2 * self.dataset
        magnetisation = torch.zeros(self.data_size)
        for i_data in range(self.data_size):
            magnetisation[i_data] = model.magnetisation(spinset[i_data]).item()
        magnetisation_data = torch.mean(magnetisation)
        return magnetisation_data


# In[ ]:


# set ising model parameters
global num_rows = 3
global num_cols = 3
beta = 4.0


# In[ ]:


# calculate true energy of Ising model
model = Ising_model(num_rows, num_cols, beta)
E_model = model.energy_expectation()
print('E_model = {}'.format(E_model))
S_model = model.entropy()
print('S_model = {}'.format(S_model))


# In[ ]:


# test if the sampler works well (how accurate it is)
data_size = 1000
sampler = IsingSampler(num_rows, num_cols, beta)
D = sampler.sample(data_size)
anal = Analyze_data(D)
E_data = anal.energy_data(model)
print('E_data = {}'.format(E_data))
S_data = anal.entropy_data()
print('S_data = {}'.format(S_data))


# In[ ]:


# set RBM parameters
nv = num_rows * num_cols
nh = 10
rbm = RBM(nv, nh)

