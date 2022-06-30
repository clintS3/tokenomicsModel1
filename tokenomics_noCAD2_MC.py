#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 18 2022

@author: clint
"""


#import some stuff
import pandas as pd
import numpy as np
import random as rd
from scipy.stats import poisson
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"


def compute_supply_reduction(decay_rate, B):
    # upcoming Bs in the next 6 months
    upcoming_Bs = [B * (decay_rate**x) for x in range(0, 25)]
    Bdiff = [(upcoming_Bs[i-1] - upcoming_Bs[i]) / upcoming_Bs[i-1] for i in range(1, len(upcoming_Bs)) ]
    df = pd.DataFrame(Bdiff)
    Bema_f = df.ewm(span=len(Bdiff)).mean()
    Bema = Bema_f.iloc[:,0].max()
    return Bema

# speculative value
def calculate_U(U_prev, momentum_U, decay_rate, B):
    sign = 1 if np.random.rand() > 0.45 else -1
    delta = U_prev * np.random.uniform(high=0.1) * sign + (np.random.uniform(high=compute_supply_reduction(decay_rate, B) * U_prev))
    new_U = U_prev + delta 
    U = momentum_U * new_U + (1 - momentum_U) * U_prev
    return U


#number of simulation timesteps
#for 13s blocktime, there are about 2.4million blocks per year.
numTimeSteps = 1000
steps = list(range(0, numTimeSteps))
myRanChoice = [1,2,3,4]

#placeholder arrays
S = np.zeros(numTimeSteps)
D = np.zeros(numTimeSteps)
Q = np.zeros(numTimeSteps)
P = np.zeros(numTimeSteps)
R = np.zeros(numTimeSteps)
K = np.zeros(numTimeSteps)
B = np.zeros(numTimeSteps)
D = np.zeros(numTimeSteps)
V = np.zeros(numTimeSteps)
U = np.zeros(numTimeSteps)
C = np.zeros(numTimeSteps)
W = np.zeros(numTimeSteps)
M = np.zeros(numTimeSteps)
Delta_D = np.zeros(numTimeSteps)
Delta_S = np.zeros(numTimeSteps)
lambda_S = np.zeros(numTimeSteps)
lambda_D = np.zeros(numTimeSteps)
departure_D = np.zeros(numTimeSteps)
departure_S = np.zeros(numTimeSteps)
numMonteCarloRuns = 5
#setup for variables to save from monte carlo runs
KK =  [[0 for x in range(numTimeSteps)] for y in range(numMonteCarloRuns)]


for mc in range(0,numMonteCarloRuns-1):
    #some constants
    u1 = 2
    u2 = 0.1
    gamma_K = float(0.5)
    alpha_C = float(0.8)
    mean_C  = float(0.5)
    sigma_C = float(0.1)
    momentum_U = float(0.7)
    epsilon_B = float(0.7)
    decay_rate = float(0.996673)
    
    #initial states
    S[0] = float(10000)
    D[0] = float(10000)
    Q[0] = float(0)
    P[0] = float(0)
    B[0] = float(69864.73)
    C[0] = float(1)
    V[0] = float(0)
    U[0] = float(1)
    K[0] = float(1)
    R[0] = float(0.0001)
    W[0] = float(float('inf')-B[0])
    M[0] = B[0]
    lambda_S[0] = float(1000)
    lambda_D[0] = float(1000)
    departure_D[0] = float(100)
    departure_S[0] = float(100)


    # SIMULATION SECTION
    #loop over timesteps for simulation
    for ii in range(1,numTimeSteps,1):
        
        # offer service stage
        Delta_D[ii] = np.random.poisson(lambda_D[ii-1])
        Delta_S[ii] = np.random.poisson(lambda_S[ii-1])
        D[ii] = D[ii-1] + Delta_D[ii] - departure_D[ii]
        S[ii] = S[ii-1] + Delta_S[ii] - departure_S[ii]
        
        # transact service stage
        # should D and S be ii or ii-1??
        P[ii] = D[ii]/S[ii]
        Q[ii] = min(D[ii],S[ii])
        U[ii] = calculate_U(U[ii-1], momentum_U, decay_rate, B[ii-1])
        new_C = (1 + np.random.normal(0, 0.1)) * C[ii-1]
        C[ii] = alpha_C * C[ii-1]+ (1 - alpha_C * new_C)
        
        # block reward
        revenue = P[ii]*Q[ii] + B[ii]
        cost    = max(1,C[ii]*Q[ii])
        V[ii]   = revenue/cost
        M[ii] = M[ii-1] + B[ii-1]
        W[ii] = W[ii-1] - B[ii-1]
        
        # tok price
        K[ii] = gamma_K*1/V[ii] + (1-gamma_K)*U[ii]
        R[ii] = K[ii]*V[ii]
        
        # update signals
        B[ii] = B[ii-1]
        lambda_S[ii] = lambda_S[ii-1]*R[ii]/R[ii-1]
        lambda_S[ii] = lambda_D[ii-1]*P[ii-1]/P[ii]
        departure_D[ii] = departure_D[ii-1]
        departure_S[ii] = departure_S[ii-1]
        
        #save monte carlo and sim steps.
        KK[mc][ii] = K[ii]
        
        

# for ii in range(1,numTimeSteps):
averages = np.zeros(numTimeSteps)
# averages = np.mean(KK[:][ii])


vals_at_timestep =  [[0 for x in range(numMonteCarloRuns)] for y in range(numTimeSteps)]

for ii in range(1,numTimeSteps):
    for mc in range(0,numMonteCarloRuns-1):
      vals_at_timestep[ii][mc] = KK[mc][ii]

averages = [ 0 for x in range(numTimeSteps)]

for ii in range(0,numTimeSteps-1):
    averages[ii] = np.mean(vals_at_timestep[ii])
  
   

fig = make_subplots(rows=1, cols=1)
row = 1
col = 1
fig.add_trace(go.Scatter(x=np.arange(1,numTimeSteps), y=averages, name='K av'), row=row,col=col)
fig.update_yaxes(title_text='K',row=row,col=col) 
fig.show()



# fig = make_subplots(rows=7, cols=2)

# row = 1
# col = 1
# fig.add_trace(go.Scatter(x=steps, y=S, name='S'), row=row,col=col)
# fig.update_yaxes(title_text='S',row=row,col=col) 

# row = 2
# fig.add_trace(go.Scatter(x=steps, y=D, name='D'), row=row,col=col)
# fig.update_yaxes(title_text='D',row=row,col=col)

# row = 3
# fig.add_trace(go.Scatter(x=steps, y=P, name='P'), row=row,col=col)
# fig.add_trace(go.Scatter(x=steps, y=C, name='C'), row=row,col=col)
# fig.update_yaxes(title_text='P & C',row=row,col=col)

# row = 4
# fig.add_trace(go.Scatter(x=steps, y=K, name='K'), row=row,col=col)
# fig.update_yaxes(title_text='K',row=row,col=col)

# row = 5
# fig.add_trace(go.Scatter(x=steps, y=R, name='R'), row=row,col=col) 
# fig.update_yaxes(title_text='R',row=row,col=col)

# row = 6
# fig.add_trace(go.Scatter(x=steps, y=1/V, name='1/V'), row=row,col=col)
# fig.add_trace(go.Scatter(x=steps, y=U, name='U'), row=row,col=col)
# fig.update_yaxes(title_text='1/V & U',row=row,col=col)


# row = 7
# fig.add_trace(go.Scatter(x=steps, y=P*K, name='P*K'), row=row,col=col)
# fig.update_yaxes(title_text='P*K',row=row,col=col) 

# row = 1
# col = 2
# fig.add_trace(go.Scatter(x=steps, y=dS, name='dS'), row=row,col=col)
# fig.add_trace(go.Scatter(x=steps, y=dD, name='dD'), row=row,col=col)
# fig.update_yaxes(title_text='dS,dS',row=row,col=col)

# row = 2
# fig.add_trace(go.Scatter(x=steps, y=lambda_s, name='lam_s'), row=row,col=col)
# fig.add_trace(go.Scatter(x=steps, y=lambda_d, name='lam_d'), row=row,col=col)
# fig.update_yaxes(title_text='lam_s,lam_d',row=row,col=col)

# row = 3
# fig.add_trace(go.Scatter(x=steps, y=C/P, name='C/P'), row=row,col=col)
# fig.add_trace(go.Scatter(x=steps, y=1/V, name='1/V'), row=row,col=col)
# fig.update_yaxes(title_text='C/P',row=row,col=col)

# row = 4
# fig.add_trace(go.Scatter(x=steps, y=P*Q, name='PQ'), row=row,col=col)
# fig.add_trace(go.Scatter(x=steps, y=B, name='B'), row=row,col=col)
# fig.update_yaxes(title_text='PQ & B',row=row,col=col)

# row = 5
# fig.add_trace(go.Scatter(x=steps, y=C*Q/(P*Q+B), name='C*Q/(P*Q+B)'), row=row,col=col)
# fig.add_trace(go.Scatter(x=steps, y=C*Q/(P*Q), name='C*Q/(P*Q)'), row=row,col=col)
# fig.add_trace(go.Scatter(x=steps, y=1/V, name='1/V'), row=row,col=col)
# fig.update_yaxes(title_text='1/V',row=row,col=col)

# row = 6
# fig.add_trace(go.Scatter(x=steps, y=V, name='V'), row=row,col=col)
# fig.update_yaxes(title_text='V',row=row,col=col)

# fig.show()

# print(np.var(P*K))

'''
#########################################
#plotting section
fig = make_subplots(rows=4, cols=1)
fig.add_trace(go.Scatter(x=steps, y=e, name='ETH in pool'), row=1,col=1)
fig.update_xaxes(title_text='Time Steps',row=1)
fig.update_yaxes(title_text='ETH',row=1)    

fig.add_trace(go.Scatter(x=steps, y=em, name='ETH in market'), row=2,col=1)
fig.update_xaxes(title_text='Time Steps',row=2)
fig.update_yaxes(title_text='ETH',row=2)  

fig.add_trace(go.Scatter(x=steps, y=k/e**2, name='TKN/ETH pool'), row=3,col=1)
fig.add_trace(go.Scatter(x=steps, y=km/em**2, name='TKN/ETH market'), row=3,col=1)
fig.update_xaxes(title_text='Time Steps',row=3)
fig.update_yaxes(title_text='TKN/ETH',row=3) 

fig.add_trace(go.Scatter(x=steps, y=e*tm/em + t, name='pool liq value in TKN'), row=4,col=1)
fig.update_xaxes(title_text='Time Steps',row=4)
fig.update_yaxes(title_text='TKN',row=4) 

fig.layout.title = '...'    
fig.show()
#########################################
'''
'''
#old implementation
#update supply and demand as a poisson process dependent 
#on previous miner incentive (supply) and customer need (demand).
if ii > 1:
    lambda_s[ii] = lambda_s[ii-1]*R[ii-1]/R[ii-2]
    lambda_d[ii] = lambda_d[ii-1]*P[ii-2]/P[ii-1]
else:
    lambda_s[ii] = lambda_s[ii-1]
    lambda_d[ii] = lambda_d[ii-1]
try:    
    dS[ii] = float(poisson.rvs(mu = lambda_s[ii], size = 1))
except:
    dS[ii] = float(poisson.rvs(mu = 0, size = 1))  
try:    
    dD[ii] = float(poisson.rvs(mu = lambda_d[ii], size = 1))
except:
    dD[ii] = float(poisson.rvs(mu = 0, size = 1))
    
    
#constant exiting of supply and demand    
Xs[ii] = Xs[ii-1]
Xd[ii] = Xd[ii-1]

#logic to manipulate supply and demand over time
# if ii >= 100 and ii <=200:
#     S[ii] = S[ii-1] + 10*dS[ii] - 5*Xs[ii] 
#     D[ii] = D[ii-1] + dD[ii] - Xd[ii]
# elif ii >= 600 and ii <=700:  
#     S[ii] = S[ii-1] + 10*dS[ii] - 20*Xs[ii] 
#     D[ii] = D[ii-1] + dD[ii] - Xd[ii]
# else:
S[ii] = S[ii-1] + dS[ii] - Xs[ii]
D[ii] = D[ii-1] + 2*dD[ii] - Xd[ii]

#price as demand/supply
P[ii] = D[ii]/S[ii]

#quantity of unit service transacted
Q[ii] = min(D[ii],S[ii])

#cost of mining one unit of service. Random value with sampling momentum
#lower c means higher random costs. range(a) = [0,1]
C[ii] = c*C[ii-1] + (1-c)*rd.normalvariate(C[ii-1],.01)
if C[ii] <= 0:
    C[ii] = C[0]

#block reward
if ii%100 == 0:
    B[ii] = B[ii-1]/2
else:
    B[ii] = B[ii-1]

#TOK earned per unit FIAT invested in system
#units are TOK/FIAT
V[ii] = (P[ii]*Q[ii] + B[ii])/(C[ii]*Q[ii]) 

#speculated price. If U goes up, R goes up. i.e. higher speculated price makes more
#peopel want to work to provide service.
#units are FIAT/TOK
U[ii] = u1*rd.normalvariate(1/V[ii],1/V[ii]*u2)

#the overall price of TOK measured in FIAT. A combination of actual value
#and speculated value. #high g means less-reliant on speculative 
#price. range(g) = [0,1]
#units are FIAT/TOK
K[ii] = g/V[ii] #+ 0*(1-g)*U[ii]  

#producer (miner) incentive to work. Good is > 1.
R[ii] = V[ii]*K[ii]
    
'''
    
    
    
    
