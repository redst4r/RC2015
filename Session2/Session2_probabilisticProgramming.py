# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:52:29 2015

@author: michi
"""

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colonies_per_spleen = dict()

colonies_per_spleen["exp1"] = np.concatenate([np.repeat(0,24),
                                      np.repeat(1,6),
                                      np.array([2,6,19,21,23,36])
                                      ])
#actually theres three experiments, the other two in table 3
colonies_per_spleen["exp2"] = np.concatenate([np.repeat(0,3),
                                      np.repeat(1,3),
                                      np.repeat(2,2),
                                      np.array([3,3, 20,20, 32])
                                      ])

colonies_per_spleen["exp3"]= np.concatenate([np.repeat(0,12),
                                      np.repeat(1,8),
                                      np.repeat(2,5),
                                      np.repeat(3,2),
                                      np.array([4,5,5, 7,8,8, 11, 13,20, 23,29,46,])
                                      ])

# throw all experiments together
data = np.concatenate(colonies_per_spleen.values())
data= colonies_per_spleen['exp1']
data = np.array([0,0,0,0,1,2,19,21,23,46])


"""
======================================
just get the posterior over N by hand!
======================================
"""
from scipy.stats import binom, poisson
p = 0.17
n = np.arange(500)

pdfArray = np.zeros((len(data), len(n)))
for i,d in enumerate(data):
    pdf = binom.pmf(d, n, p)
    pdf = pdf/pdf.sum()
    pdfArray[i,:] = pdf

plt.plot(pdfArray.T)

#what the distribution then (taking into account the uncertainty in the estimate)
marginalStemCellDist = pdfArray.sum(0)/pdfArray.sum()
plt.plot(marginalStemCellDist)

#compared to the one where we only use the MAP
#create the same kind of matrix but just with delta peaks at the MAP
MAP_locations = np.argmax(pdfArray,axis=1)
pdfArray_MAPonly = np.zeros((len(data), len(n)))
for i, lo in enumerate(MAP_locations):
    pdfArray_MAPonly[i,lo] = 1

marginalStemCellDist_MAPonly = pdfArray_MAPonly.sum(0)/pdfArray_MAPonly.sum()
plt.plot(marginalStemCellDist_MAPonly)

""" CDFS """
CDF_uncertain = np.cumsum(marginalStemCellDist)
CDF_MAP = np.cumsum(marginalStemCellDist_MAPonly)

plt.figure()
plt.plot(n, CDF_uncertain)
plt.plot(n, CDF_MAP)

CDF_poisson = poisson.cdf(n,mu=31)
plt.plot(n, CDF_poisson)



"---------------------------------------------"
"THE FULL GAMMA MODEL"
"---------------------------------------------"
data = data[data!=0] # somehow the zeros are not handled well, MCMC craps out as soon as we hit 0

with pm.Model() as model:
    #the fraction of stem celsl creating a clone, estimated to be 0.17+- 0.02
    myTheta = pm.Beta('theta', alpha=17, beta=83)#uncretainty roughly covered by this prior

    # the number o stem cells per clone, according to the model, this is a
    # Gamma distribution, which we acutally want to fit
    myAlpha = pm.Uniform('myAlpha', lower=0.0001, upper=10)
    myBeta = pm.Uniform('myBeta', lower=0.00001, upper=0.1)  # this is 1/stats.gamma(alpha, BETA!), i.e. if the true para = stats.gamma(beta)=160, we use 1/160
    N = pm.Gamma('N',alpha=myAlpha, beta=myBeta, shape=len(data))

    # the number of colonies formed. thats what we observe
    k = pm.Binomial(name='k', n=N, p=myTheta, observed=data)

with model:
    start = pm.find_MAP()
with model:
    # somehow the MAP doesnt really work with the discrete N variables
    # just initiailzie at k/0.17

    #alpha and beta are initialized at the MLE fits without uncertainty propagation for ease
    start = {'theta': 0.17, 'myAlpha': 0.11, 'myBeta': 1/168.0, 'N': 1+np.floor(np.array(data/0.17))}

# NUTS doesnt work with discrete data apparently
#    step = pm.NUTS(vars=[myAlpha,myBeta])
#    trace = pm.sample(1000, step, start)
#

    # neat, can combine samplers:
    #https://stackoverflow.com/questions/21352696/difficulities-on-pymc3-vs-pymc2-when-discrete-variables-are-involved
    step1 = pm.Metropolis(vars=[myAlpha,myBeta, myTheta]) #, N

    # the unknown number of stem cells  is very different across observations
    # which makes it hard to get them with a single proposal
    # hence we supply a variance vector, such that high cell counts also have wider proposals
    covMat = 0.5 * (data+1)
    step2 = pm.Metropolis(vars=[N], S= covMat)

    trace = pm.sample(200000, [step1,step2], start)

pm.traceplot(trace[10000::500])

"some plotting"
from pandas.tools.plotting import scatter_matrix
from pandas import DataFrame

traj = np.vstack((trace['theta'],trace['myAlpha'], trace['myBeta'])).T[10000::500,:]
df = DataFrame(traj, columns=['fraction', 'alpha', 'beta'])
plt.figure()
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

plt.figure()
plt.plot(traj)

#====================================
# posterior predictive (kind of)
from scipy import stats
def ecdf(x):
    """ calculate the empirical distribution function of data x """
    sortedX=np.sort( x )
    yvals=np.arange(len(sortedX))/float(len(sortedX))
    return sortedX, yvals

plt.figure()
x,y = ecdf(data/0.17)
plt.plot(x,y, linewidth=5)
for i in range(traj.shape[0]):
    tmpTheta, tmpAlpha, tmpBeta = traj[i,:]

    currentGamma = stats.gamma(tmpAlpha, scale= 1/tmpBeta)
    yTmp = currentGamma.cdf(x)
    plt.plot(x,yTmp,'g',alpha=0.1)

plt.plot(x,y, 'b', linewidth=5)
plt.xlabel('#CFUs')
plt.ylabel('CDF')
plt.legend(['observed', 'posterior predictive'], loc=4)
