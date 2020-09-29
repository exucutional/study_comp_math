#!/usr/bin/env python
# coding: utf-8

# # Lab 02
# ## Solving the system of linear equations by Gauss and Seidel method
# ### Konks Eric, Б01-818
# II.10.6(б)

# $$\begin{cases}b_1x_1+c_1x_2 = f_1\\a_2x_1+b_2x_2+c_2x_3=f_2\\a_3x_2+b_3x_3+c_3x_4=f_3\\...\\a_nx_{n-1}+b_nx_n+c_nx_{n+1}=f_n\\p_1x_1+p_2x_2+...+p_{n-1}x_{n-1}+p_nx_n+p_{n+1}x_{n+1}=f_{n+1}\end{cases}$$

# $$n=9; b_1=1; c_1=0; f_1=1$$

# $$a_i=c_i=1; b_i=-2; p_i=2; f_i= 2/i^2; i=2, 3, ..., n-1$$

# $$f_{n+1} = -n/3; p_1=p_{n+1}=1$$

# In[1]:


import unittest
import logging
import numpy as np
import pandas as pd


# In[2]:


#logging.basicConfig(level=logging.INFO)


# In[3]:


class Gauss:
    
    @staticmethod
    def __swap_line(m, l1, l2):
        for i in range(len(m[l1])):
            (m[l1][i], m[l2][i]) = (m[l2][i], m[l1][i])
    
    @classmethod
    def __forward(cls, m):
        for i in range(len(m)):
            if m[i][i] == 0:
                j = i
                while m[j][i] == 0:
                    j = j + 1
                    
                if j < len(m):
                    cls.__swap_line(m, i, j)

            if m[i][i] != 0:
                linec = np.copy(m[i])
                for j in range(i, len(linec)):
                    linec[j] = linec[j] / m[i][i]

                for ii in range(i+1, len(m)):
                    tmp = m[ii][i]
                    for jj in range(len(m[ii])):
                        m[ii][jj] = m[ii][jj] - linec[jj] * tmp

    @staticmethod
    def __backward(m):
        result = [0 for i in range(len(m))] + [-1]
        for i in reversed(range(len(m))):
            for j in range(len(m[i])):
                if i != j:
                    result[i] = result[i] - result[j] * m[i][j] / m[i][i]
                
        return result[:-1]
    
    @classmethod
    def solve(cls, m):
        m_c = np.copy(m)
        cls.__forward(m_c)
        return cls.__backward(m_c)


# In[4]:


class Seidel:
    
    @staticmethod
    def solve(m, init, delta):
        log = logging.getLogger("seidel")
        isComp = False
        res = init + [-1.]
        iteration = 0
        while not isComp:
            res_next = np.copy(res)
            for i in range(len(m)):
                res_next[i] = 0
                for j in range(len(m[i])):
                    if i != j:
                        res_next[i] = res_next[i] - res_next[j] * m[i][j] / m[i][i]

            isComp = np.sqrt(sum((res_next[i] - res[i])**2 for i in range(len(res)))) < delta
            res = res_next
            log.debug(f"Iter[{iteration}]: {res}")
            iteration = iteration + 1
            
        return res[:-1]


# In[5]:


class GaussAndSeidelTest(unittest.TestCase):
    
    def equal(self, res, exp, delta):
        result = True
        for i in range(len(res)):
            if np.abs(res[i] - exp[i]) > delta:
                result = False
            
        return result
    
    def setUp(self):
        self.matrixes = [
           [[2., 3., -4.],
            [3., 8., 1.]],
            
           [[2., 3., -1., 9.],
            [1., -2., 1., 3.],
            [1., 0., 2., 2.]],
            
           [[7., 9., -10., -6., -1., -6., -4., 8., -5., 122.],
            [-2., -2., 1., -6., 1., 8., -6., -3., -5., 34.],
            [9., 6., 6., -3., 5., -9., -9., -7., 5., 9.],
            [-1., 2., -8., -8., 9., 1., 8., 1., -9., 84.],
            [-1., 2., 1., 8., 3., 2., -5., 8., 8., -220.],
            [-5., -3., 0., 6., -1., 5., 1., -8., -2., 35.],
            [-6., -6., 0., 9., 2., -3., 3., -3., 6., -50.],
            [-2., -3., -2., -8., 5., -10., -7., 0., -2., 107.],
            [5., 1., -2., 3., -2., 9., -2., 2., 6., -81.]],
            
           [[10., 1., 1., 12.],
            [2., 10., 1., 13.],
            [2., 2., 10., 14.]]
        ]
        
        self.exp_results = [
            [-5, 2],
            [4, 0, -1],
            [4, -5, -10, -4, -7, -6, -1, -9, -8],
            [1, 1, 1]
        ]
        
        self.init = [
            [1.0, 1.0],
            [3.0, 0.0, -0.5],
            [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1, -1.0, -1.0],
            [0., 0., 0.]
        ]
        
        self.isSeidel = [True, False, False, True]
        
        self.delta = 10**-10
    
    def test_cases(self): 
        for i in range(len(self.matrixes)):
            log = logging.getLogger(f"gauss_test_case_{i}\n")
            res_gauss = Gauss.solve(self.matrixes[i])
            isEqual_gauss = self.equal(res_gauss, self.exp_results[i], self.delta)
            log_gauss = pd.DataFrame({"Result": res_gauss, "Expected": self.exp_results[i]})
            log.info(log_gauss)
            if self.isSeidel[i]:
                log = logging.getLogger(f"seidel_test_case_{i}\n")
                res_seidel = Seidel.solve(self.matrixes[i], self.init[i], self.delta)
                isEqual_seidel = self.equal(res_seidel, self.exp_results[i], self.delta)
                log_seidel = pd.DataFrame({"Result": res_seidel, "Expected": self.exp_results[i]})
                log.info(log_seidel)

            self.assertTrue(isEqual_gauss)
            self.assertTrue(isEqual_seidel)


# In[6]:


unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[7]:


log = logging.getLogger("task")

n = 9

A = np.array([np.array([0. for _ in range(n+2)]) for _ in range(n+1)])

b1 = 1
c1 = 0
f1 = 1

A[0][0] = b1
A[0][1] = c1
A[0][n+1] = f1

for i in range(1, n):
    fi = 2 / (i+1)**2
    ai = 1
    ci = 1
    bi = -2
    (A[i][i-1], A[i][i], A[i][i+1]) = (ai, bi, ci)
    A[i][n+1] = fi
    
A[n][0] = 1
A[n][n] = 1
A[n][n+1] = -n / 3.
    
for i in range(1, n):
    pi = 2
    A[n][i] = pi

print(f"{pd.DataFrame(A)}\n")

init = [0. for _ in range(n+1)]
delta = 1/10**10
gauss_res = Gauss.solve(A)
gauss_residual = [A[i][n+1] - sum(A[i][j]*gauss_res[j] for j in range(n+1)) for i in range(n+1)]
seidel_res = Seidel.solve(A, init, delta)
seidel_residual = [A[i][n+1] - sum(A[i][j]*seidel_res[j] for j in range(n+1)) for i in range(n+1)]
print(f"Seidel method stopping criteria: ||x_k+1 - x_k|| < {delta}\n")
print(f"Condition number μ = {np.linalg.cond(A[:,:-1])}\n")
print(pd.DataFrame({"Eigenvalues λ": np.linalg.eigvals(A[:,:-1])}))
print(pd.DataFrame({"Gauss": gauss_res, "Seidel": seidel_res}))
print(pd.DataFrame({"Gauss_residual": gauss_residual, "Seidel_residual": seidel_residual}))

