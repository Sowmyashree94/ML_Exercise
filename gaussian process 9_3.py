# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 20:11:45 2018

@author: sowmyashree
"""
# x_star are sample data points
# x is the given data


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

n = 100
gamma_values = [2,1] # lambda = sigma power 2
sigma_values = [0,0.1]
l = 0.2 #global variable
data = np.array([[-0.5,0.3], [0.5,-0.1]])


#returns the entries of the kernel matrix
def k_value(x, x_prime, gamma):
    kVal = np.exp(-(abs(x-x_prime)/l)**gamma)
    return kVal



#Kernel matrix
def k_matrix(x1,x2,gamma):
    K = np.zeros((x1.shape[0],x2.shape[0]))
    for i, xi in enumerate(x1):
        for j,xj in enumerate(x2):
            K[i,j] = k_value(xi,xj,gamma)
    return K



def covar_matrix(x_star, x, gamma, sigma):
    K_x = k_matrix(x,x,gamma)
    K_x = K_x + (sigma**2)*np.eye(K_x.shape[0]) #add noise
    
    K_x_star = k_matrix(x, x_star,gamma)
    
    K_x_star_trans = K_x_star.T
    
    K_x_star_star = k_matrix(x_star, x_star,gamma)
    K_x_star_star = K_x_star_star + (sigma**2)*np.eye(K_x_star_star.shape[0])
    
    covar_K = np.bmat([[K_x, K_x_star],[K_x_star_trans, K_x_star_star]])
    return K_x, K_x_star, K_x_star_trans, K_x_star_star, covar_K



def mean_value(K_x_star, K_x, sigma,y):
    temp = np.linalg.inv(K_x + (sigma**2)*np.eye(K_x.shape[0]))
    mean_val = np.linalg.multi_dot([K_x_star_trans , temp, y])
    return mean_val



def covariance_value(K_x, K_x_star, K_x_star_trans, K_x_star_star, sigma):
    temp = np.linalg.inv(K_x + ((sigma**2)*np.eye(K_x.shape[0])))
    temp1 = np.linalg.multi_dot([K_x_star_trans, temp , K_x_star])
    covar_val = K_x_star_star - temp1 
#    + (sigma**2)*np.eye(K_x_star_star.shape[0])
    return covar_val




x_star = np.linspace(-2.8, 2.8, n).reshape(-1,1)

count = 1

for gamma in gamma_values :
    for sigma in sigma_values : 
        K_x, K_x_star, K_x_star_trans , K_x_star_star, covar_K = covar_matrix(x_star, data[:,0],gamma, sigma)
        mean = mean_value(K_x_star, K_x, sigma, data[:,1])
        covariance = covariance_value(K_x, K_x_star, K_x_star_trans, K_x_star_star, sigma)

        #check pdf page 9
        std_deviation = np.sqrt(covariance.diagonal()) 
        f_posterior_plus_std = mean + std_deviation
        f_posterior_minus_std = mean - std_deviation
    

        fontP = FontProperties()
        fontP.set_size('small')
        plt.figure(count)
        plt.clf()
        plt.plot(data[:,0],data[:,1], 'rx')
        plt.plot(x_star, mean, label = 'mean')
        plt.plot(x_star, f_posterior_plus_std, label = 'mean + standard deviation')
        plt.plot(x_star, f_posterior_minus_std , label = 'mean - standard deviation')
        plt.title('Gaussian process posterior with gamma = %f and sigma = %f'%(gamma,sigma))
        plt.legend(bbox_to_anchor=(-0.05, 1.15), loc=2,ncol=1, mode="none", borderaxespad=0.)
        count = count+1