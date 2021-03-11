# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 00:05:59 2021

@author: Jeff Scanlon
Andrew ID: jscanlo2
"""

from gurobipy import *
import numpy as np
import math
import pandas as pd

num_districts = 36
num_regions = 9
num_dist_per_region = 4
regions = range(num_regions)
districts = range(num_districts)

# A big number to help formulate big-M constraint
M = 1000
hollow_votes = np.array([3,0,0,0,5,0,1,1,0,0,0,8,0,0,4,6,0,7,2,0,2,6,0,4,0,9,5,7,3,0,0,0,0,0,8,0])
solid_votes  = np.array([0,9,1,6,0,1,0,0,7,0,5,0,7,4,0,0,5,0,0,3,0,0,2,0,8,0,0,0,0,6,0,4,3,8,0,2])

# 36 by 36 matrix to indicate whether two cells are adjacent
is_neighbor = np.zeros([num_districts, num_districts])
for i in districts:
    for j in districts:
        if np.abs(np.mod(i,6) - np.mod(j,6)) == 1.0 and math.floor(i/6) == math.floor(j/6):
            is_neighbor[i,j] = 1
        if np.mod(i,6) == np.mod(j,6) and np.abs(math.floor(i/6) - math.floor(j/6))== 1.0:
            is_neighbor[i,j] = 1

m = Model()
# district-region assignment variable
x = m.addVars(districts, regions, vtype=GRB.BINARY)
#x[i,k] = 1 if i is district i is assigned to region k, 0 otherwise

# region winning (by a certain margin) indicator
y = m.addVars(regions, vtype=GRB.BINARY)
# y[k] = 1 if Solid wins region k by 4 votes or more, 0 otherwise

# you may need additional variables for the contiguity constraints
u = m.addVars(districts, districts, regions, vtype=GRB.BINARY)
#u[i,j,k] = 1 if districts i and j in region k are contiguous, 0 otherwise

# Each district must be assigned and one and only one region
for i in districts:
    m.addConstr(sum(x[i,k] for k in regions) == 1)
    
# Each region must consist of exactly 4 districts
for k in regions:
    m.addConstr(sum(x[i,k] for i in districts) == 4)

# Y = 0 if Solid doesn't win by 4 or more votes
for k in regions:
    m.addConstr((sum(solid_votes[i]*x[i,k] for i in districts) - sum(hollow_votes[i]*x[i,k] for i in districts)) <= 3 + M*(y[k]))

# Y = 1 if Solid does win by 4 or more votes
for k in regions:
    m.addConstr((sum(solid_votes[i]*x[i,k] for i in districts) - sum(hollow_votes[i]*x[i,k] for i in districts)) >= 4 - M*(1-y[k]))


# Contiguity constraints
for k in regions:
    for i in districts:
        for j in districts:
            m.addConstr(u[i,j,k]  <= x[i,k]) # district i has to be in region k for u = 1, otherwise 0
            m.addConstr(u[i,j,k] <= x[j,k]) # district j also has to be in region k for u = 1, otherwise 0
            m.addConstr(u[i,j,k] <= is_neighbor[i,j]) # districts i and j have to be contiguous for u = 1, otherwise 0
 

# Contiguity constraints: Each region should have at least 6 pairs of neighbors (accounting for double counting of contiguous relationships)
for k in regions:
    m.addConstr(sum(u[i,j,k] for i in districts for j in districts) >= 6)


# Set Objective Function
WinningRegions = sum(y[k] for k in regions)

m.setObjective(WinningRegions, GRB.MAXIMIZE)

m.optimize()
       
/ 

    
#Model details
print('The way to divide the regions into 9 regions')
rows = []
for i in districts:
    rows.append(sum(x[i,k].x *(k+1) for k in regions))
    results = [rows[:6],rows[6:12],rows[12:18],rows[18:24],rows[24:30],rows[30:]]
    resultspandas = pd.DataFrame(results)
print(resultspandas)