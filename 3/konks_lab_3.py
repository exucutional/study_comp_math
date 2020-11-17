#!/usr/bin/env python
# coding: utf-8

# # Lab 02
# 
# ## Solving a system of nonlinear equations
# 
# ### Konks Eric, Б01-818
# 
# IV.12.7.д

# $$\begin{cases} x^7 - 5x^2y^4 + 1510 = 0 \\ y^3 - 3x^4y - 105 = 0 \end{cases}$$

# $$\begin{cases} x_{n+1} = \sqrt{\frac{x_n^7 + 1510}{5y_n^4}} \\ y_{n+1} = \sqrt[3]{3x_{n}^4y_{n}+105} \end{cases}$$

# $$J=\begin{pmatrix}7x^6-10xy^4 & -20x^2y^3\\-12x^3y & 3y^2-3x^4\end{pmatrix}$$

# In[1]:


import unittest
import logging
import numpy as np
import pandas as pd


# In[2]:


#logging.basicConfig(level=logging.DEBUG)


# In[3]:


class FPI:
    def __init__(self, f_vec):
        self.__f_vec = f_vec
        self.iter = 0
        self.log = logging.getLogger("FPI")
    
    def __is_stop(self, next_x, cur_x, q, delta):
        if next_x == cur_x:
            return False
        
        if sum(np.abs((next_x[i] - cur_x[i])) for i in range(len(cur_x))) <= delta * (1 - q):
            return True
        
        return False
    
    def solve(self, init_x, q, delta):
        cur_x = init_x
        next_x = init_x
        while not self.__is_stop(next_x, cur_x, q, delta):
            cur_x = next_x
            next_x = cur_x[:]
            for i in range(len(self.__f_vec)):
                next_x[i] = self.__f_vec[i](cur_x)
              
            self.log.debug(f"Iter[{self.iter}]: Init: {cur_x} Next: {next_x}")
            self.iter = self.iter + 1
        
        return next_x      


# In[4]:


class Newton:
    def __init__(self, f_vec, J):
        self.__f_vec = f_vec
        self.__J = J
        self.iter = 0
        self.log = logging.getLogger("Newton")
    
    def __J_mul_f(self, x, i):
        return sum(self.__f_vec[j](x) * self.__J[i][j](x) for j in range(len(self.__f_vec)))
    
    def __is_stop(self, next_x, cur_x, M2, m1, delta):
        if next_x == cur_x:
            return False
        if sum(np.abs(next_x[i] - cur_x[i]) for i in range(len(cur_x))) < np.sqrt(2*delta*m1/M2):
            return True
        
        return False
    
    def solve(self, init_x, M2, m1, delta):
        self.iter = 0
        cur_x = init_x
        next_x = init_x
        while not self.__is_stop(next_x, cur_x, M2, m1, delta):
            cur_x = next_x
            next_x = cur_x[:]
            for i in range(len(self.__f_vec)):
                next_x[i] = cur_x[i] - self.__J_mul_f(cur_x, i)
            
            self.log.debug(f"Iter[{self.iter}]: Init: {cur_x} Next: {next_x}")
            self.iter = self.iter + 1
            
        return next_x


# In[5]:


def fpi_f1(x):
    return np.sqrt((x[0]**7 + 1510)/(5 * (x[1]**4)))

def fpi_f2(x):
    return np.cbrt(3*(x[0]**4)*x[1] + 105)

fpi = FPI([fpi_f1, fpi_f2])


# In[6]:


def newton_f1(x):
    return x[0]**7-5*(x[0]**2)*(x[1]**4)+1510

def newton_f2(x):
    return x[1]**3-3*(x[0]**4)*x[1]-105

def J00(x):
    return 7*(x[0]**6)-10*x[0]*(x[1]**4)

def J01(x):
    return -20*(x[0]**2)*(x[1]**3)

def J10(x):
    return -12*(x[0]**3)*x[1]

def J11(x):
    return 3*(x[1]**2) - 3*(x[0]**4)

def J(x):
    return [[J00(x), J01(x)], [J10(x), J11(x)]]

def J00_inv(x):
    return J11(x)/(J00(x)*J11(x)-J10(x)*J01(x))

def J01_inv(x):
    return - J01(x)/(J00(x)*J11(x)-J10(x)*J01(x))

def J10_inv(x):
    return - J10(x)/(J00(x)*J11(x)-J10(x)*J01(x))

def J11_inv(x):
    return J00(x)/(J00(x)*J11(x)-J10(x)*J01(x))

J_inv = [[J00_inv, J01_inv], [J10_inv, J11_inv]]
newton = Newton([newton_f1, newton_f2], J_inv)


# In[7]:


log = logging.getLogger()
x_init_vec_fpi = [[1,5], [3, -4], [-1, 5]]
x_init_vec_newton = [[1,5], [3, -4], [-1, 5], [-4, 0], [-2, -2]]
delta = 10**-5
q = 0.5
m1 = 1
M2 = 1
fpi_results = []
fpi_iterations = []
newton_results = []
newton_iterations = []
for x in x_init_vec_fpi:
    fpi_results.append(fpi.solve(x, q, delta))
    fpi_iterations.append(fpi.iter)
    
for x in x_init_vec_newton:
    newton_results.append(newton.solve(x, M2, m1, delta))
    newton_iterations.append(newton.iter)


# In[8]:


fpi_dt = pd.DataFrame({"Начальное приближение": x_init_vec_fpi, "Результат": fpi_results, "Итераций": fpi_iterations})
newton_dt = pd.DataFrame({"Начальное приближение": x_init_vec_newton, "Результат": newton_results, "Итераций": newton_iterations})
print("Метод простых итераций")
print(fpi_dt)
print("\nМетод Ньютона")
print(newton_dt)

