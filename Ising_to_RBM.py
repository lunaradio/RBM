import torch
import itertools
import numpy as np
import math
import matplotlib.pyplot as plt


def decimal_to_binary_tensor(value, width=0):
    string = format(value, '0{}b'.format(width))
    binary = [0 if c == '0' else 1 for c in string]
    return torch.tensor(binary, dtype=torch.float)



class Ising_model():
    
    # initiate the size of 2d lattice, the (inverse-)temperature, interactions
    def __init__(self, num_rows, num_cols):
        self.num_rows, self.num_cols = num_rows, num_cols
        self.num_spins = num_rows * num_cols
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
        H = - torch.matmul(s, torch.matmul(J, s)) / 2
        return H
    
    def magnetisation(self, spin_state):
        m = torch.mean(spin_state)
        return m
    
    def partition_function(self, beta):
        num_state = 2**(self.num_spins)
        Z = 0.
        for i_state in range(num_state):
            binary = decimal_to_binary_tensor(i_state, width=self.num_spins)
            spin_state = 2 * binary - 1
            H = self.hamiltonian(spin_state)
            Z += torch.exp(- beta * H)
        return Z
    
    # calculate the expectation value of the Energy of Ising model with Boltzmann distribution
    def energy_expectation(self, beta):
        num_state = 2**(self.num_spins)
        Z = self.partition_function(beta)
        E_exp = 0.
        for i_state in range(num_state):
            binary = decimal_to_binary_tensor(i_state, width=self.num_spins)
            spin_state = 2 * binary - 1
            H = self.hamiltonian(spin_state)
            E_exp += H * torch.exp(- beta * H)
        return E_exp / Z
    
    def magnetisation_expectation(self, beta):
        num_state = 2**(self.num_spins)
        Z = self.partition_function(beta)
        m_exp = 0.
        for i_state in range(num_state):
            binary = decimal_to_binary_tensor(i_state, width=self.num_spins)
            spin_state = 2 * binary - 1
            m = self.magnetisation(spin_state)
            H = self.hamiltonian(spin_state)
            m_exp += m * m * torch.exp(- beta * H)
        return m_exp / Z
    
    def entropy(self, beta):
        Z = self.partition_function(beta)
        return beta * self.energy_expectation(beta) + torch.log(Z)




class RBM():
    
    # Initiate the sizes of visible and hidden layers, and RBM parameters(weights, biases)
    def __init__(self, num_rows, num_cols, beta):
        self.num_rows, self.num_cols, self.beta = num_rows, num_cols, beta
        self.num_v = num_rows * num_cols
        self.num_h = 2 * num_rows * num_cols - num_rows - num_cols
        self.W = torch.zeros(self.num_v, self.num_h)
        self.a = torch.zeros(self.num_v)
        self.b = torch.zeros(self.num_h)
        self.model = Ising_model(num_rows, num_cols)
        self.J = self.model.J
        self.beta = beta
    
    def label_edges(self, i, j):
        J = self.J
        num_horizontal_edges = (self.num_cols - 1) * self.num_rows
        
        if J[i, j] == 1:
            if i - j == self.num_cols:
                e = num_horizontal_edges + j
            elif i - j == - self.num_cols:
                e = num_horizontal_edges + i
            elif i - j == 1:
                e = (self.num_cols - 1) * int(i / self.num_cols) + i % self.num_cols - 1
            elif i - j == -1:
                e = (self.num_cols - 1) * int(i / self.num_cols) + i % self.num_cols
            return e
        elif J[i, j] == 0:
            print("site i and j are not on the same edge")
        else:
            print("invalid i and j for given num_rows and num_cols")
    
    def get_weights(self, visible):
        J = self.J
        beta = self.beta
        self.W = torch.zeros(self.num_v, self.num_h)
        for i in range(self.num_v):
            for j in range(i, self.num_v):
                if J[i, j] != 0:
                    e = self.label_edges(i, j)
                    self.W[i, e] = 2 * np.arccosh(np.exp(2 * beta * J[i, j]))
                    self.W[j, e] = 2 * np.arccosh(np.exp(2 * beta * J[i, j]))

    def get_biases_a(self):
        J = self.J
        beta = self.beta
        self.a = torch.zeros(self.num_v)
        for i in range(self.num_v):
            for e in range(self.num_h):
                self.a[i] -= 0.5 * self.W[i, e]

    def get_biases_b(self):
        J = self.J
        beta = self.beta
        self.b = torch.zeros(self.num_h)
        for e in range(self.num_h):
            for i in range(self.num_v):
                self.b[e] -= 0.5 * self.W[i, e]
        
    def to_hidden(self, visible):
        b = self.b
        vW = torch.matmul(visible, self.W)
        return torch.sigmoid(vW + b)

    def to_visible(self, hidden):
        a = self.a
        Wh = torch.matmul(self.W, hidden)
        return torch.sigmoid(Wh + a)




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
        spinset = 2 * self.dataset - 1
        energy = torch.zeros(self.data_size)
        for i_data in range(self.data_size):
            energy[i_data] = model.hamiltonian(spinset[i_data]).item()
        energy_data = torch.mean(energy)
        return energy_data
    
    def magnetisation_data(self):
        spinset = 2 * self.dataset - 1
        magnetisation = torch.zeros(self.data_size)
        for i_data in range(self.data_size):
            magnetisation[i_data] = torch.mean(spinset[i_data]).item()
        magnetisation_data = torch.mean(magnetisation)
        return magnetisation_data




class Ising_sampler():
    
    # initiate the size of a dataset, and create a sampled dataset
    def __init__(self, sample_size, num_rows, num_cols, beta):
        self.sample_size = sample_size
        self.num_rows, self.num_cols = num_rows, num_cols
        self.beta = beta
    
    # randomly generate one sample of an spin state as 2d tensor
    def gen_random_sample(self, p = 0.5):
        probs = torch.rand(self.num_rows, self.num_cols)
        sample = torch.where(probs < p, torch.zeros(1), torch.ones(1))
        return sample
    
    # create a training data set by using Metropolis algorithm
    def gibbs_sampling(self, v_initial, num_steps):
        rbm = RBM(self.num_rows, self.num_cols, self.beta)
        num_burn = 0
        # initialize an ising sample
        v_next = v_initial
        # save the sample of the visible layer to the dataset
        self.sample = v_next.unsqueeze(0)
        
        for _ in range(self.sample_size - 1):
            for _ in range(num_steps):
                v_current = v_next
                # calculate RBM parameters W and a from a given visible layer by Ising-RBM mapping
                rbm.get_weights(v_current)
                rbm.get_biases_a()
                rbm.get_biases_b()

                pr_h = rbm.to_hidden(v_current)
                h_next = torch.bernoulli(pr_h)
                pr_v = rbm.to_visible(h_next)
                v_next = torch.bernoulli(pr_v)

            self.sample = torch.cat((self.sample, v_next.unsqueeze(0)), dim = 0)

        return self.sample




# set the size and temperature of Ising model, sample size, and initial visible layer
num_rows = 4
num_cols = 4
beta = 0.3
sample_size = 1000
sampler = Ising_sampler(sample_size, num_rows, num_cols, beta)
v_initial = torch.ones(num_rows, num_cols).view(-1)

# make a graph of m_square of the data in terms of the number of gibbs steps up to 10
m_square = torch.zeros(10)
for num_steps in range(10):
    sample = sampler.gibbs_sampling(v_initial, num_steps)
    anal = Analyze_data(sample)
    m_square[num_steps] = anal.magnetisation_data()

x = np.arange(10)
y = m_square.numpy()

plt.plot(x, y, 'r', label='m_squre')
plt.xlabel('the number of gibbs steps')
plt.legend(loc='best')
plt.show()



"""
beta = torch.arange(0.0, 1.0, 0.01)
model = Ising_model(num_rows, num_cols)
m_square = model.magnetisation_expectation(beta)
E_model = model.energy_expectation(beta)

x = beta.numpy()
y1 = m_square.numpy()
y2 = E_model.numpy()

plt.plot(x, y1, 'r', label='m_squre')
plt.xlabel('beta')
plt.legend(loc='best')
plt.show()

plt.plot(x, y2, 'b', label='E')
plt.xlabel('beta')
plt.legend(loc='best')
plt.show()

"""



