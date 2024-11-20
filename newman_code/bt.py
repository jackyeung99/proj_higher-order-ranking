#!/usr/bin/env python3

# Maximum-likelihood/MAP estimation of Bradley-Terry rankings using the
# algorithm of M. Newman, JMLR (2023)
#
# Written by Mark Newman  8 JUNE 2023

# Imports
from sys import argv,stderr
from numpy import zeros,ones,empty,loadtxt,max,prod,copy,log
import numpy as np 
import random

# Parameters
TARGET = 1e-6   # Target accuracy
MAP = True      # Set to True for MAP estimation, False for maximum likelihood


# Read data
def readdata(filename):
    data = loadtxt(filename,int,delimiter=",")
    n = max(data) + 1
    m = len(data)
    return data,n,m


# Make adjacency lists
def mklists(data,n,m):
    outdeg = zeros(n,int)
    indeg = zeros(n,int)
    outadj = empty(n,list)
    inadj = empty(n,list)
    for i in range(n):
        outadj[i] = []
        inadj[i] = []
    for q in range(m):
        i,j = data[q]
        outadj[i].append(j)
        inadj[j].append(i)
        outdeg[i] += 1
        indeg[j] += 1
    return outdeg,indeg,outadj,inadj

def random_number_from_logistic():
    return 1.0 / np.random.rand() - 1.0
# Main program

# Read data
filename = argv[1]
data,n,m = readdata(filename)
outdeg,indeg,outadj,inadj = mklists(data,n,m)
print("Read data set with n =",n,"items and m =",m,"interactions",file=stderr)
if MAP: print("Calculating maximum a posteriori ranking...",file=stderr)
else:   print("Calculating maximum-likelihood ranking...",file=stderr)

# Main loop
# pi = ones(n,float)
# pi = np.array([random_number_from_logistic() for _ in range(n)])
pi = np.array([np.random.rand() for _ in range(n)])
print(pi)
delta = 1.0
r = 0
while delta>TARGET:

    r += 1
    print("Iteration",r,end="\r",file=stderr)

    oldpi = copy(pi)
    for i in range(n):

        # Calculate numerator and denominator
        if MAP: tsum = bsum = 1/(pi[i]+1)
        else:   tsum = bsum = 0.0
        for q in range(outdeg[i]):
            j = outadj[i][q]
            tsum += pi[j]/(pi[i]+pi[j])
        for q in range(indeg[i]):
            j = inadj[i][q]
            bsum += 1/(pi[i]+pi[j])

        # Update the strength parameter
        pi[i] = tsum/bsum

    # For maximum likelihood estimation, normalize the pi's.  This step is
    # not needed for MAP estimates.
    if not MAP:
        norm = prod(pi)**(1/n)
        pi /= norm

    # Find the largest change in pi/(pi+1) for any i
    newp = pi/(pi+1)
    oldp = oldpi/(oldpi+1)
    delta = max(abs(newp-oldp))

# Save the results
filebase = filename.split(".")[0]
outfile = filebase+".pi"
fp = open(outfile,"w")
print("ITEM,PI,S",file=fp)
for i in range(n): print(i,pi[i],log(pi[i]),sep=",",file=fp)
fp.close()

# Print summary
print("Iterations to converge =",r,file=stderr)
print("Results written to CSV file",outfile,file=stderr)
print("Done.",file=stderr)
