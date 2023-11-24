#!/usr/bin/env python
# coding: utf-8

# In[29]:


import random
import matplotlib.pyplot as plt
import math


# ## Bernoulli distribution

# - Statistical meaning: A discrete probability distribution that gives only two possible results in a trial (aka an experiment).

# - Real-life examples: 
#     - Flip a coin: the probability of head versus tail
#     - Charity asks for donations: the probability of donated versus ignored

# - Graphs:
# ![image-2.png](attachment:image-2.png)

# In[36]:


def generate_prn():
    return random.random()


# In[37]:


# generate x bernoulli random variates i.e. trials with probability p

def generate_bernoulli_rv(p, x):
    outputs = []
    prns = []
    
    for i in range(x):
        prn = generate_prn()
        prns.append(prn)
        
        if prn < 1-p:
            outputs.append(0)
        else:
            outputs.append(1)
    
    return outputs, prns


# In[127]:


def draw_bernoulli_rv(p, x):
    rv_outputs, prns = generate_bernoulli_rv(p, x)
    
    x_axis = ['0', '1']
    y_axis = [rv_outputs.count(0), rv_outputs.count(1)]
    
    plt.title(label='p = '+str(p)+', x = '+str(x))
    plt.bar(x=x_axis, height=y_axis)
    
    return rv_outputs


# In[54]:


bernoulli_p = 0.2


# In[141]:


# small sample size
bernoulli_rv_outputs = draw_bernoulli_rv(p=bernoulli_p, x=1)


# In[142]:


bernoulli_rv_outputs


# In[130]:


# small sample size
bernoulli_size = 10

bernoulli_rv_outputs = draw_bernoulli_rv(p=bernoulli_p, x=bernoulli_size)


# In[131]:


bernoulli_rv_outputs


# In[132]:


# big sample size
bernoulli_size = 1000

bernoulli_rv_outputs = draw_bernoulli_rv(p=bernoulli_p, x=bernoulli_size)


# In[133]:


bernoulli_rv_outputs


# ## Binomial distribution

# - Statistical meaning: For an experiment (trial) that gives only 'Success' and 'Failure' outcome, the number of ‘Success’ in a sequence of *n* experiments (aka trials).

# - Real-life examples: 
#     - Flip a coin: the number of heads and tails in X times (aka trial, experiments)  
#     - Charity asks for donations: the number donations in a month

# - Graphs:
# ![image-3.png](attachment:image-3.png)

# In[111]:


# generate x binomial random variates of bernoulli trials with probability p

def generate_binomial_rv(p, x):
    outputs = []
    
    for i in range(x):
        # the number of successes and failures out of Bernoulli trials
        bernoulli_output = generate_bernoulli_rv(p, x)[0] 
        # the number of successes (=1)
        outputs.append(bernoulli_output.count(1)) 
        
    return outputs


# In[136]:


def draw_binomial_rv(p, x):
    rv_outputs = generate_binomial_rv(p, x)
    
    fig, ax = plt.subplots(1,2)

    ax[0].set_title(label='p = '+str(p)+', x = '+str(x))
    ax[0].set_ylabel('count')
    ax[0].hist(x=rv_outputs)
    
    ax[1].set_title(label='p = '+str(p)+', x = '+str(x))
    ax[1].set_ylabel('probablity')
    values = []
    probs = []
    total_num = len(rv_outputs)
    for i in rv_outputs:
        values.append(i)
        probs.append(rv_outputs.count(i)/total_num)
    ax[1].bar(x=values, height=probs)
    
    return rv_outputs


# In[118]:


binomial_p = 0.7


# In[137]:


# small sample size
binomial_size = 20

binomial_rv_outputs = draw_binomial_rv(p=binomial_p, x=binomial_size)


# In[138]:


binomial_rv_outputs


# In[139]:


# big sample size
binomial_size = 1000

binomial_rv_outputs = draw_binomial_rv(p=binomial_p, x=binomial_size)


# In[91]:


draw_binomial_rv(p=binomial_p, x=binomial_size, y_axis='probability')


# In[140]:


binomial_rv_outputs


# ## Geometric distribution

# - Statistical meaning: The probability of having *X* Bernoulli trials with probability *p* until a Success occurs, i.e. the first success.

# - Real-life examples: 
#     - Job search: how many applications until receiving an offer
#     - Defect products: how many items produced until the first defective unit

# - Graphs:
# ![image-4.png](attachment:image-4.png)

# In[221]:


def geometric_pmf(p):
    outputs = []
    failure_count = 0
    
    while failure_count < 10:
        outputs.append((1-p)**failure_count*p)
        failure_count += 1
        
    return outputs, failure_count


# In[222]:


def draw_geometric_distribution(p):
    rv_outputs, trial_count = geometric_pmf(p)

    plt.title(label='p = '+str(p))
    plt.ylabel('probablity')
    values = [i for i in range(trial_count)]
    plt.plot(values, rv_outputs)


# In[223]:


geometric_p = 0.2
draw_geometric_distribution(p=geometric_p)


# In[224]:


geometric_p = 0.8
draw_geometric_distribution(p=geometric_p)


# In[225]:


draw_geometric_distribution(p=0.2)
draw_geometric_distribution(p=0.6)
draw_geometric_distribution(p=0.8)


# In[280]:


# generate x geometric random variate of bernoulli trials with probability p

def generate_geometric_rv(p, x):
    outputs = []
    trial_counts = []
    
    for i in range(x):
        bernoulli_outcome = 0
        trial_count = 0
        
        while bernoulli_outcome != 1: # keep going until a Success occurs
            bernoulli_outcome = generate_bernoulli_rv(p, 1)[0][0]
            trial_count += 1

        outputs.append((1-p)**(trial_count-1)*p)
        trial_counts.append(trial_count) 
        
    return outputs, trial_counts


# In[290]:


def draw_geometric_rv(p, x):
    rv_outputs, trial_counts = generate_geometric_rv(p, x)

    plt.title(label='p = '+str(p))
    values = [i for i in trial_counts]
    plt.bar(values, rv_outputs)
    
    return trial_counts


# In[284]:


geometric_p = 0.2


# In[293]:


# small sample size
geometric_size = 10

geometric_rv_outputs = draw_geometric_rv(p=geometric_p, x=geometric_size)


# In[294]:


geometric_rv_outputs


# In[295]:


# big sample size
geometric_size = 1000

geometric_rv_outputs = draw_geometric_rv(p=geometric_p, x=geometric_size)


# In[296]:


geometric_rv_outputs


# ## Poisson distribution

# - Statistical meaning: The probability of an event happening a certain number of times *k* within a given interval of time or space. 

# - Real-life examples: 
#     - Call Center: the probability that the call center receives more than 5 phone calls during the noon, given the average 3 calls per hour during that time period.
#     - Broadcast: the probablity that the news reporter says "uh" more than three times during a broadcast, given the average 2 "uh" per broadcast.

# - Graphs:
# ![image-6.png](attachment:image-6.png)

# In[338]:


# generate x poisson random variate in one time unit

def generate_poisson_rv(l, x):
    outputs = []
    
    for i in range(x):
        prn = generate_prn()
        time_unit = 0 
        cmf = 0
        while prn > cmf:
            pmf = math.e**-l*l**time_unit/math.factorial(time_unit)
            cmf = cmf + pmf
            time_unit += 1
        outputs.append(time_unit)    
        
    return outputs


# In[372]:


def draw_poisson_rv(l, x):
    rv_outputs = generate_poisson_rv(l, x)

    plt.title(label='lambda = '+str(l))
    
    values = []
    counts = []
    for i in set(rv_outputs):
        values.append(i)
        counts.append(rv_outputs.count(i))

    plt.plot(values, counts)
    
    return rv_outputs


# In[378]:


poisson_lambda = 2


# In[380]:


# small sample size
lambda_size = 10

poisson_rv_outputs = draw_poisson_rv(l=poisson_lambda, x=lambda_size)


# In[381]:


poisson_rv_outputs


# In[382]:


# big sample size
lambda_size = 1000

poisson_rv_outputs = draw_poisson_rv(l=poisson_lambda, x=lambda_size)


# In[383]:


poisson_rv_outputs


# In[385]:


lambda_size = 1000

poisson_lambda = 2
poisson_rv_outputs = draw_poisson_rv(l=poisson_lambda, x=lambda_size)

poisson_lambda = 4
poisson_rv_outputs = draw_poisson_rv(l=poisson_lambda, x=lambda_size)


# In[ ]:




