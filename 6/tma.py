#!/usr/bin/env python
# coding: utf-8

# # Lab 06
# 
# ## Solving boundary value problem with tridiagonal matrix algorithm
# 
# ### Konks Eric, Б01-818
# 
# Task - 5

# $$\frac{d}{dx}[k(x)\frac{du}{dx}]-q(x)u=-f(x)$$

# $$k(0)u_x(0)=u(0)$$

# $$-k(1)u_x(1)=u(1)$$

# $$k(x)=x^2+1\ \ \ q(x)=x\ \ \ f(x)=e^{-x}$$

# In[1]:


import unittest
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#logging.basicConfig(level=logging.DEBUG)


# In[3]:


class TMA:
    def __init__(self):
        self.log = logging.getLogger("TMA")
    
    def calc_coeff(self, k, q, f, d1, e1, d2, e2, h, x, limit):
        a = 0
        b = 0
        c = 0
        d = 0
        if abs(x - limit[0]) < h/10:
            a = k(x)
            b = -k(x)-d1*h
            c = 0
            d = -e1*h
        elif abs(x - limit[1]) < h/10:
            a = 0
            b = -k(x)-d2*h
            c = k(x)
            d = -e2*h
        else:
            a = k(x)
            b = -2*k(x)-q(x)*(h**2)
            c = k(x)
            d = -f(x)*(h**2)
            
        return (a, b, c, d)
        
    
    def solve(self, k, q, f, d1, e1, d2, e2, acc, N, limit):
        h = (limit[1]-limit[0])/(N-1)
        x = []
        u = []
        a = []
        b = []
        c = []
        d = []
        alpha = []
        beta = []
        for l in range(N):
            (al,bl,cl,dl) = self.calc_coeff(k, q, f, d1, e1, d2, e2, h, limit[0]+h*l, limit)
            a.append(al)
            b.append(bl)
            c.append(cl)
            d.append(dl)
            x.append(limit[0]+l*h)
            if l == 0:
                alpha.append(-al/bl)
                beta.append(dl/bl)
            else:
                alpha.append(-al/(bl+cl*alpha[l-1]))
                beta.append((dl-cl*beta[l-1])/(bl+cl*alpha[l-1]))
            
            self.log.debug(f"[{x[l]}]: {al} {bl} {cl} {dl} {alpha[l]} {beta[l]}")

        u.append((d[N-1]-c[N-1]*beta[N-2])/(b[N-1]+c[N-1]*alpha[N-2]))
        
        for l in reversed(range(N-1)):
            u.insert(0,alpha[l]*u[0]+beta[l])
            
        return (x, u)


# In[4]:


class TMATest(unittest.TestCase):
    def equal(self, res, exp, acc):
        result = True
        for i in range(len(res)):
            if np.abs(res[i] - exp[i]) > acc:
                result = False
            
        return result
    
    def test_cases(self):
        k = lambda x: 1.25
        q = lambda x: 0.5
        f = lambda x: 1/np.sqrt(np.exp(1))
        d1 = 1
        e1 = 0
        d2 = 1
        e2 = 0
        acc = 0.0001
        N = 10000
        limit = (0, 1)
        h = (limit[1]-limit[0])/(N-1)
        tma = TMA()
        tma_res = tma.solve(k, q, f, d1, e1, d2, e2, acc, N, limit)

        lambda1 = np.sqrt(q(0)/k(0))
        lambda2 = -np.sqrt(q(0)/k(0))
        
        C1_1 = (k(0)*lambda2+d2)*(d1*f(0)-e1*q(0))*np.exp(lambda2)+(k(0)*lambda2-d1)*(d2*f(0)-e2*q(0))
        C1_2 = q(0)*((k(0)*lambda1-d1)*(k(0)*lambda2+d2)*np.exp(lambda2)-(k(0)*lambda2-d1)*(k(0)*lambda1+d2)*np.exp(lambda1))
        C1 = C1_1/C1_2
        C2_1 = (k(0)*lambda1+d2)*(d1*f(0)-e1*q(0))*np.exp(lambda1)+(k(0)*lambda1-d1)*(d2*f(0)-e2*q(0))
        C2_2 = q(0)*((k(0)*lambda2-d1)*(k(0)*lambda1+d2)*np.exp(lambda1)-(k(0)*lambda2+d2)*(k(0)*lambda1-d1)*np.exp(lambda2))
        C2=C2_1/C2_2
        
        dir_x = []
        dir_u = []
            
        for l in range(N):
            dir_x.append(limit[0]+l*h)
            dir_u.append(C1*np.exp(lambda1*(limit[0]+l*h))+C2*np.exp(lambda2*(limit[0]+l*h))+f(0)/q(0))
        
        log = logging.getLogger(f"TMATest\n")
        isEqual = self.equal(tma_res[1], dir_u, acc)
        if not isEqual:
            df = pd.DataFrame({"x": tma_res[0], "u": tma_res[1]})    
            log_res = pd.DataFrame({"x": tma_res[0], "Result u": tma_res[1], "Expected u": dir_u})
            log.error(log_res)
        
        self.assertTrue(isEqual)


# In[5]:


unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[6]:


k = lambda x: x**2+1
q = lambda x: x
f = lambda x: np.sqrt(np.exp(-x))
d1 = 1
e1 = 0
d2 = 1
e2 = 0
acc = 0.0001
N = 10001
N_res = 11
limit = (0, 1)
tma = TMA()
tma_res = tma.solve(k, q, f, d1, e1, d2, e2, acc, N, limit)
res = (tma_res[0][0::int(N/N_res)][:N_res], tma_res[1][0::int(N/N_res)][:N_res])
#res = tma_res
df = pd.DataFrame({"x": res[0], "u": res[1]})
print(df)

