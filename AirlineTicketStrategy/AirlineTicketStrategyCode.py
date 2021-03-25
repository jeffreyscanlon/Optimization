# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 22:18:17 2021

@author: jeffr
"""

from gurobipy import *
import numpy as np
import csv
import os
import pandas as pd
import gurobipy as gp
from numpy import genfromtxt
import matplotlib.pyplot as plt

# Read in CSV files
PATH = os.getcwd()
os.chdir('C:\\Users\\jeffr\\Documents\\Homework6_DABP')

D = genfromtxt(PATH + '\\Pb1_D_stochastic.csv', delimiter=',') 
P = genfromtxt(PATH + '\\Pb1_prob.csv', delimiter=',')



# Set up indices
scenarios = range(1000)
ticketgroups = range(4)


# Stochastic Programming Model
m = Model()

# First Stage Decision Variables
X = m.addVars(ticketgroups, lb=0.0, vtype=GRB.INTEGER)

# Second Stage Decision Variables
Y = m.addVars(ticketgroups, scenarios, lb=0.0, vtype=GRB.INTEGER)


# Objective Function
Revenue = gp.LinExpr()
for s in scenarios:
    Revenue += (400 * Y[0, s] * P[s]) + (500 * Y[1, s] * P[s]) + (800 * Y[2, s] * P[s]) + (1000 * Y[3, s] * P[s])

m.setObjective(Revenue, GRB.MAXIMIZE)


# Constraints
for t in ticketgroups:
    for s in scenarios:
        m.addConstr(Y[t, s] <= D[t, s])
        

for t in ticketgroups:
    for s in scenarios:
        m.addConstr(X[t] >= Y[t,s])
        # m.addConstr(Y2[t, s] >= 0) This constraint is accounted for with "lb = 0" in decision variable construction
        

m.addConstr(X[0] + (1.2 * X[1]) + (1.5 * X[2]) + (2 * X[3]) <= 190)


m.optimize()

Revenue.getValue()

for t in ticketgroups:
        print(X[t].x)

print("Number of Economy Seats:", X[0].x)
print("Number of Economy+ Seats:", X[1].x)
print("Number of Business Seats:", X[2].x)
print("Number of First Class Seats:", X[3].x)
        
# Average Demand
print(D.mean())
print("Average Demands:")
print("Economy:", D[0].mean())
print("Economy+:", D[1].mean())
print("Business:", D[2].mean())
print("First Class:", D[3].mean())

# Average Demand
print("Average Demand, Scaled by Seat Size")
print("Economy:", D[0].mean())
print("Economy+:", D[1].mean()/1.2)
print("Business:", D[2].mean()/1.5)
print("First Class:", D[3].mean()/2)

totdemands = []
for i in scenarios:
    totdemands.append(sum(D[:,i]))

count1 = 0
count2 = 0
for s in scenarios:
    if Y[0,s].x < 164 or Y[1,s].x < 15 or Y[2,s].x < 4 or Y[3,s].x < 1:
        count1 += 1
    else:
        count2 += 1

print("Number of scenarios underbooked:", count1)
print("Number of scenarios fully booked:", count2)

Sold1 = []
D1 = []
Sold2 = []
D2 = []
Sold3 = []
D3 = []
Sold4 = []
D4 = []
for s in scenarios:
    Sold1.append(Y[0,s].x)
    D1.append(D[0,s])
    Sold2.append(Y[1,s].x)
    D2.append(D[1,s])
    Sold3.append(Y[2,s].x)
    D3.append(D[2,s])
    Sold4.append(Y[3,s].x)
    D4.append(D[3,s])

plt.hist(Sold1)
plt.hist(D1, alpha = 0.7)
plt.xlabel("Number of Economy Seats")
plt.ylabel("Number of Scenarios")
plt.title("Distribution of Economy Seats Demanded (Orange) and Sold (Blue)")


plt.hist(Sold2)
plt.hist(D2, alpha = 0.7)
plt.xlabel("Number of Economy+ Seats")
plt.ylabel("Number of Scenarios")
plt.title("Distribution of Economy+ Seats Demanded (Orange) and Sold (Blue)")

plt.hist(Sold3)
plt.hist(D3, alpha = 0.7)
plt.xlabel("Number of Business Seats")
plt.ylabel("Number of Scenarios")
plt.title("Distribution of Business Seats Demanded (Orange) and Sold (Blue)")

plt.hist(Sold4)
plt.hist(D4, alpha = 0.7)
plt.xlabel("Number of First Class Seats")
plt.ylabel("Number of Scenarios")
plt.title("Distribution of First Class Seats Demanded (Orange) and Sold (Blue)")

    

count3 = 0
count4 = 0
for s in scenarios:
    if Y[0,s].x < D[0,s] or Y[1,s].x < D[1,s] or Y[2,s].x < D[2,s] or Y[3,s].x < D[3,s]:
        count3 += 1
    else:
        count4 += 1

print("Number of scenarios where tickets sold is less than demand:", count3)
print("Number of scenarios where tickets sold equal demand:", count4)
        

# Model 2

#Add Parameters
fares = np.array([[400,500,800,1000], [400,500,600,700], [400,420,600,700]])
f_probs = np.array([0.4, 0.3, 0.3])

m2 = Model()

# First Stage Decision Variables
X2 = m2.addVars(ticketgroups, lb=0, vtype=GRB.INTEGER)

# Second Stage Decision Variables
Y2 = m2.addVars(ticketgroups, scenarios, lb=0, vtype=GRB.INTEGER)

# Objective Function
Revenue2 = gp.LinExpr()
for s in scenarios:
    for f in range(3):
        Revenue2 += (fares[f][0] * f_probs[f] * Y2[0, s] * P[s]) \
        + (fares[f][1] * f_probs[f] * Y2[1, s] * P[s])\
        + (fares[f][2] * f_probs[f] * Y2[2, s] * P[s])\
        + (fares[f][3] * f_probs[f] * Y2[3, s] * P[s])

m2.setObjective(Revenue2, GRB.MAXIMIZE)


# Constraints
for t in ticketgroups:
    for s in scenarios:
        m2.addConstr(Y2[t, s] <= D[t, s])
      # m2.addConstr(Y2[t, s] >= 0) This constraint is accounted for with "lb = 0" in decision variable construction
        

for t in ticketgroups:
    for s in scenarios:
        m2.addConstr(X2[t] >= Y2[t,s])
        

m2.addConstr(X2[0] + (1.2 * X2[1]) + (1.5 * X2[2]) + (2 * X2[3]) <= 190)


m2.optimize()

Revenue2.getValue()

print("Number of Economy Seats:", X2[0].x)
print("Number of Economy+ Seats:", X2[1].x)
print("Number of Business Seats:", X2[2].x)
print("Number of First Class Seats:", X2[3].x)

