# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:54:05 2021

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

energy = genfromtxt(PATH + '/Pb1_energy.csv', delimiter=',') 
mapping = genfromtxt(PATH + '/Pb1_mapping.csv', delimiter=',')
price = genfromtxt(PATH + '/Pb1_price.csv', delimiter=',')
composition = genfromtxt(PATH + '/Pb1_composition.csv', delimiter=',')

# Generate parameters
requirements = np.array([20000, 400, 7, 6.5, 0.57, 20, 0.7, 1.1, 0.050, 0.0005])
groupmins = np.array([1.1, 0.3, 35, 4, 0.9, 0.5])
groupmaxs = np.array([25.1, 8.7, 75, 33, 10.2, 8.5])

composition[1, 0]

# Generate Indices
n_fooditems = range(1007)
n_nutrients = range(10)
n_foodgroups = range(6)

m = Model()

# Decision Variables
# X[i] is number of grams of food item i prescribed by the diet
x = m.addVars(n_fooditems, lb=0.0, vtype=GRB.CONTINUOUS)

# Objective 1
# Minimizing daily energy intake by each child
TotalEnergy = sum(x[i]*energy[i]/100 for i in n_fooditems)

# Objective 2
# Minimizing total daily cost of the diet
TotalCost = sum(x[i]*price[i]/100 for i in n_fooditems)

lst = []
alpha_values = [0, 0.001, 0.005, 0.01, 0.015, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1]        
outcomes = np.zeros([len(alpha_values), 2])

for i in range(len(alpha_values)):
    
    alpha = alpha_values[i]
    
    # Objective Function for Weight-Based Approach
    m.setObjective((alpha*TotalEnergy) + ((1-alpha)*TotalCost), GRB.MINIMIZE)
    
    # Objective Function for Goal Programming Approach
   # m.setObjective(TotalEnergy, GRB.MINIMIZE)
    
    # Constraints
    # Must meet WHO Nutrient Requirement
    for j in n_nutrients:
        m.addConstr(sum(x[i]*composition[i,j]/100 for i in n_fooditems) >= requirements[j])
    
    # Minimum Proportion Constraint
    for k in n_foodgroups:
        m.addConstr(sum(x[i]*mapping[i,k]*energy[i] for i in n_fooditems) >= sum(x[i]*energy[i]/100 for i in n_fooditems) * groupmins[k])
    
    # Maximum Proportion Constraint
    for k in n_foodgroups:
        m.addConstr(sum(x[i]*mapping[i,k]*energy[i]for i in n_fooditems) <= sum(x[i]*energy[i]/100 for i in n_fooditems) * groupmaxs[k])
        
    # Additional WHO cost constraint of $0.50 per day
    #m.addConstr(sum(x[i]*price[i]/100 for i in n_fooditems) <= 0.5)
      
    m.optimize()
    
    print("Total energy:", TotalEnergy.getValue())
    print("Total cost:", TotalCost.getValue())
    
    outcomes[i, 0] = TotalEnergy.getValue()
    outcomes[i, 1] = TotalCost.getValue()
    
    for l in n_fooditems:
        if x[l].x > 0:
            lst.append((i, l, x[l].x))
    
plt.scatter(outcomes[:,1], outcomes[:,0])
plt.xlabel("Total Cost per Child ($/day)")
plt.ylabel("Total Energy per Child (Kcal/day)")
plt.title("Pareto-Optimal Frontier")



### The following code is not necessary to Grade.
### I just wanted to check my work to make sure my constraints were set properly.

# There is probably a much cleaner way to do this
# But this is my attempt to make sure my constraints are working correctly
lst

lst0 = []
for i in lst:
    if i[0] == 0:
        lst0.append(i[1:3])
        
lind = []
for l in lst0:
    lind.append(l[0])

for l in lind:
    print(mapping[l])

en = []
for l in lind:
    en.append(energy[l])
    
num = []
for n in lst0:
    num.append(n[1])

t = list(zip(num, en))

do = []
for i in t:
    print(i[0]*i[1])
    do.append(i[0]*i[1])
    
a = sum(do)

for i in do:
    print(i/a)
    
# Minimum Proportion is binding constraint for Vegetables, Fruits, and Proteins for this optimal solution.
# Maximum Proportion is binding constraint for Grains and Oils for this optimal solution.
# For this optimal solution, Dairy is not pushing up against either Minimum or Maximum Proportion constraint.


# Check for Nutrition Constraint
com = []
for l in lind:
    com.append(composition[l])

prot = []
for p in com:
    prot.append(p[0])

protein = list(zip(num, prot))

prototal = []
for p in protein:
    prototal.append(p[0]*p[1])

sum(prototal)
    
# Protein = 2000000 mg per 100 days
# which equals 20000 per 1 day
# Which equals 20 grams per day
    


    





