# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 13:49:15 2021

Homework Assignment #3

@author: Jeff Scanlon
Andrew ID: jscanlo2
"""


import os
import numpy as np
import pandas as pd
from numpy import genfromtxt
import gurobipy as gp
from gurobipy import GRB
from plotnine import *

PATH = os.getcwd()

# Import parameters
shelters = genfromtxt(PATH + '/Pb2_shelters.csv', delimiter=',') # 40*1
areas = genfromtxt(PATH + '/Pb2_areas.csv', delimiter=',') # 200*1
distances = genfromtxt(PATH + '/distances.csv', delimiter=',') # 200*40

n_shelters = 40
n_areas = 200


### Model Object

m = gp.Model()

### Decision Variables ###
# Shelter location decision variable, taking value 0 or 1
# x[j] == 1 if shelter j is built, 0 otherwise
x = m.addVars(n_shelters, vtype=GRB.BINARY)

# Area decision variable
# y[i,j] = if residential area i is assigned to j, 0 otherwise
y = m.addVars(n_areas, n_shelters, vtype=GRB.BINARY)

### Setting up optimization direction: min ###
# The objective is to minimize the total distance traveled to shelters

TotDistance = gp.LinExpr()
for i in range(n_areas):
    for j in range(n_shelters):
        TotDistance += y[i,j]*distances[i,j]*areas[i]


m.setObjective(TotDistance, GRB.MINIMIZE)

### Setting Constraints

# Constraint 1: Build 10 shelters
m.addConstr(sum(x[j] for j in range(n_shelters)) <= 10)

# Constraint 2: All areas are assigned a shelter
for i in range(n_areas):
        m.addConstr(sum(y[i,j] for j in range(n_shelters)) == 1)
        
# Constraint 3: No area assignment to a location without a shelter
for i in range(n_areas):
    for j in range(n_shelters):
        m.addConstr(y[i,j] <= x[j])
                
#Constraint 4: Cannot exceed shleter capacity
    for j in range(n_shelters):
        m.addConstr((sum(areas[i]*y[i,j] for i in range(n_areas)) <= shelters[j]))

m.optimize()

m.objVal

TotDistance.getValue()

### Plotting Histogram
shelterLocs = np.zeros(shape=(200,40))
for i in range(n_areas):
    for j in range(n_shelters):
        shelterLocs[i,j] = y[i,j].x
        
        
# Create a list of coordinates to index "distances" matrix
trueShelters = np.where(shelterLocs == 1)
trueShelters = list(zip(trueShelters[0], trueShelters[1]))
coord = []
for s in trueShelters:
    coord.append(s)
coord



# Use list of coordinates to index "distances" and retrieve distance values
distance_list = []
for c in range(len(coord)):
    distance_list.append(distances[coord[c]])

# Use "areas" to add one distance observation per individual in area i.
distance_individ = []
for i in range(200):
    for x in range(int(areas[i])):
        distance_individ.append(distance_list[i])
        
# Use ggplot from plotnine library for plotting       
distance_individ = pd.DataFrame(distance_individ, columns = ['Distance'])
ggplot(distance_individ, aes(x='Distance')) + geom_histogram(bins = 30) + labs(y = "Count", title = "Model 1: Distribution of distances across all individuals")

print(np.mean(distance_individ))
print(np.min(distance_individ))
print(np.max(distance_individ))

#print(np.mean(distance_list))
#print(min(distance_list))
#print(max(distance_list))

# Use ggplot from plotnine library for plotting
#distance_list = pd.DataFrame(distance_list, columns = ['Distance'])
#ggplot(distance_list, aes(x='Distance')) + geom_histogram(bins = 30)


    


### Second Model
m2 = gp.Model()

### Decision Variables ###
# Shelter location decision variable, taking value 0 or 1
# x[j] == 1 if shelter j is built, 0 otherwise
x = m2.addVars(n_shelters, vtype=GRB.BINARY)

# Area decision variable
# y[i,j] = if residential area i is assigned to j, 0 otherwise
y = m2.addVars(n_areas, n_shelters, vtype=GRB.BINARY)

### Setting up optimization direction: min ###
# The objective is to minimize the total distance traveled to shelters

MaxDistance = m2.addVars(1, lb=0.0)

m2.setObjective(MaxDistance[0], GRB.MINIMIZE)

### Setting Constraints
# Constraint 1: Require all distances to be less than or equal the max distance variable
for i in range(n_areas):
    for j in range(n_shelters):
        m2.addConstr(y[i,j]*distances[i,j] <= MaxDistance[0])

# Constraint 2: Build 10 shelters
m2.addConstr(sum(x[j] for j in range(n_shelters)) <= 10)

# Constraint 3: All areas are assigned a shelter
for i in range(n_areas):
        m2.addConstr(sum(y[i,j] for j in range(n_shelters)) == 1)
        
# Constraint 4: No area assigned to a location without a shelter
for i in range(n_areas):
    for j in range(n_shelters):
        m2.addConstr(y[i,j] <= x[j])
                
#Constraint 5: Cannot exceed shleter capacity
    for j in range(n_shelters):
        m2.addConstr((sum(areas[i]*y[i,j] for i in range(n_areas)) <= shelters[j]))

m2.optimize()

m2.objVal


### Plotting Histogram
shelterLocs2 = np.zeros(shape=(200,40))
for i in range(n_areas):
    for j in range(n_shelters):
        shelterLocs2[i,j] = y[i,j].x

# Create list of coordinates to index "distances"
trueShelters2 = np.where(shelterLocs2 == 1)
trueShelters2 = list(zip(trueShelters2[0], trueShelters2[1]))
coord2 = []
for s2 in trueShelters2:
    coord2.append(s2)

# Use list of coordinates to index "distances" and retrieve distance values
distance_list2 = []
for c2 in range(len(coord2)):
    distance_list2.append(distances[coord2[c2]])
distance_list2
    
# Use "areas" to add one distance observation per individual in area i.
distance_individ2 = []
for i in range(200):
    for x in range(int(areas[i])):
        distance_individ2.append(distance_list2[i])
        
# Use ggplot from plotnine library for plotting       
distance_individ2 = pd.DataFrame(distance_individ2, columns = ['Distance'])
ggplot(distance_individ2, aes(x='Distance')) + geom_histogram(bins = 45) + labs(y = "Count", title = "Model 2: Distribution of distances across all individuals")

print(np.mean(distance_individ2))
print(np.min(distance_individ2))
print(np.max(distance_individ2))
  
#print(np.mean(distance_list2))
#print(min(distance_list2))
#print(max(distance_list2))    

# Use ggplot from plotnine library for plotting
distance_list2 = pd.DataFrame(distance_list2, columns = ['Distance'])
ggplot(distance_list2, aes(x='Distance')) + geom_histogram(bins = 30)




