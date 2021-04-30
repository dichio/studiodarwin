#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:54:39 2020
Last revised : Wer Mar 17 2021
@author: Vito Dichio


Modified cristallball.py adapted to automatically classify a QLE-like phase, a NRC-phase. It is possible to explore the parameter space, for instance in the N, \sigma directions (as done in the paper).
"""

# %% Import modules ##########
           
import os, sys, time, gc
import numpy as np
sys.path.append('/***/***/***/Darwin_Studio') # specify the location of your working directory
import FFPopSim as pop
import matplotlib.pyplot as plt
from matplotlib import colors
if __name__ == "__main__": 
    print(sys.argv)

# %% Initialization & Evolution ##########
start_time = time.time()
np_rng_seed = 1 ; np.random.seed(np_rng_seed)     # Fix the ran_seed 

def cm(x):# As in cristalball.py
    return x*0.3937
char_size = 7
label_size = 6
def RFM(mu,sigma,L): #As in cristalball.py
    """
    Random Fitness Model: initializes an LxL F_matrix of epistatic fitness (symmetric, diagonal=0) with L(L-1)/2 gaussian numbers $\sim N(\mu,\sigma). Generated data have a sigma_true whose value has a relative error < 1/1000. (mu_true, sigma_true) in output.
    """
    sigma_true = 10.0
    mu_true = 0.0
    while (abs(sigma_true - sigma)/sigma > 0.001) :
        Fij = np.random.normal(mu, sigma, size=(L*(L-1)/2)) 
        mu_true = np.mean(Fij)
        sigma_true = np.sqrt(np.var(Fij))
    F_matrix = np.zeros((L,L))
    k = 0
    for i in range(1,L):
        for j in range(0,i):
            F_matrix[i][j] = Fij[k+j]
        k += i 
    F_matrix += F_matrix.T
    return [mu_true, sigma_true, F_matrix]
def declare_initial(params,i): #As in cristalball.py
    """
    Print of initial parameters of the system.
    """
    print('*--*--*--*--*--*--*--*--*')
    print('|  N = %d' %params[1])
    print('|  sigma = %.5f' %params[9])
    print('|  iter = %i' %i)
    print('*--*--*--*--*--*--*--*--*')
    print('')
def FFPopSim_initialization(L,N,mu,r,rho,sigma,spin,cln_sizes,F): #As in cristalball.py
    """
    FFPopSim is initialized with input parameters
    """
    hpop.recombination_model = pop.CROSSOVERS # FREE_RECOMBINATION/SINGLE_CROSSOVER/CROSSOVER
    hpop.carrying_capacity = N # expected size of the population
    hpop.outcrossing_rate = r # probability of sexual reproduction per generation
    hpop.crossover_rate = rho # crossover rate per site per generation
    hpop.mutation_rate = mu
    hpop.circular = True # circular genotype
    hpop.set_genotypes(spin, cln_sizes) # Input genotypes
    hpop.clear_fitness() # add pairwise fitness coefficients to fitness function
    for i in range(L):
        for j in range(L):
            hpop.add_fitness_coefficient(F[i][j], [i, j]) 
def ev_freqs(params,t,chi,locus): #As in cristalball.py
    """
    Evolution of allele frequencies. Specify locus. If all_loci then set locus = 'all'.
    """
    if locus >= L and not locus == 'all': 
        print('This locus does not exist!')
    bwidth = 0.5
    figsizex = cm(17)
    figsizey = cm(5)
    fig = plt.figure(figsize=(figsizex, figsizey))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwidth)
    ax.spines['top'].set_linewidth(bwidth)
    ax.spines['left'].set_linewidth(bwidth)
    ax.spines['right'].set_linewidth(bwidth) 
    cmap = plt.get_cmap('prism')
    if locus == 'all':
       for loc in range(L):
           color = cmap(float(loc) / L)  # cm.cool(locus)
           plt.plot(t, chi[:, loc], ls='-', lw=0.15, c=color)
    else :
       color = cmap(float(locus) / L)  # cm.cool(locus)
       plt.plot(t, chi[:, locus], ls='-', lw=0.15, c=color, label='$\chi_{%i}(t)$' %(locus+1) )    
       plt.legend(loc='upper center',fontsize=char_size)
    plt.xlabel('t',fontsize=char_size)
    ax.text(.5,1.05,'$\{\chi_i(t)\}_{i=1}^L$', horizontalalignment='center', transform=ax.transAxes,fontsize=char_size)
    plt.xlim([0,params[2]])
    plt.tick_params(labelsize=label_size)
    ax.ticklabel_format(style='scientific',axis='x',scilimits=(0,0))
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(label_size)
    #plt.title('L=%i' %params[0] +' ' + 'N=%i' %params[1] +' '+ '$\mu$=%.2f' %params[4] +' '+ ' $r$=%.2f'%r +' '+ ' $\sigma_{tr}$=%.3f'%params[9], fontsize=char_size)
    plt.savefig('Plots/ev_f.png')
    #plt.show()
    fig.tight_layout()
    plt.close(fig) 
def ev_fitness_stats(t,fit_mean,fit_var):  #As in cristalball.py
    fig = plt.figure(figsize=(cm(17),cm(5)))
    ax = plt.gca()
    plt.plot(t, fit_mean, lw=.15, label='fitness mean')
    plt.plot(t, np.sqrt(fit_var), lw=0.15, label='fitness standard deviation')
    plt.legend(loc=2,fontsize=char_size)
    plt.xlim([0,params[2]])
    #plt.ylim([,])
    plt.tick_params(labelsize=char_size,rotation=0)
    ax.ticklabel_format(style='scientific',axis='x',scilimits=(0,0))
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(label_size)
    plt.xlabel('t',fontsize=char_size)
    plt.tick_params(labelsize=label_size)
    #plt.title('L=%i' %params[0] +' ' + 'N=%i' %params[1] +' '+ '$\mu$=%.2f' %params[4] +' '+ ' $r$=%.2f'%r +' '+ ' $\sigma_{tr}$=%.3f'%params[9], fontsize=char_size)
    plt.savefig('Plots/ev_fs.png')
    #plt.show()
    plt.close(fig)   
def classifier(t,fit_mean,params,j):    
    """
    Classifier QLE/NRC. The sharpest transition QLE/ is observed for the fitness mean (fit_mean), which is considered here. A threshold is empirically set between the aprrox. value of fit_mean at QLE and that at NRC. if the mean value of fit_mean over the past window time steps is less than threshold, we set phase=0 (QLE), otherwise phase=1 (NRC).
    """
    window = 100
    threshold = 0.4
    phase = []
    for i in range(window,len(fit_mean)):
        if np.mean(fit_mean[(i-window):i]) > threshold :
            phase.append(1)
        else :
            phase.append(0)
    plot_ev = True
    if plot_ev:
        plt.figure(figsize=(10,2))
        plt.step(t[window-1:-1],phase[:], ls='-', lw=2.,color='red')
        plt.ylim([0,1])
        plt.title('QLE-phase(0) vs NRC-phase(1)' + '\n' 'N = %i, $\sigma$ = %.5f, iter = %i' %(params[1],params[7],j), fontsize=15)
        plt.xlabel('time',fontsize=15)
        plt.yticks([0,1])
        plt.xlim([0,t[-1]])
        plt.savefig('Plots/classify.png')
        #plt.show()
    return phase
def av(x):
    if len(x) != 0:
        return float(sum(x))/len(x)
    else:
        return x    
def run_average(t,phase,params,plot_run_av=True): # check dimensions!!
    window = 100
    window_av = 900
    run_av = []
    for i in range(1,len(phase)):
        run_av.append(av(phase[0:i]))
    print('Running average at T = %i is %.5f' %(t[-1],run_av[-1])) 
    if plot_run_av:
        plt.figure(figsize=(10,2))
        plt.step(t[window_av+window+1:],run_av[window_av:], ls='-', lw=1.,color='red')
        plt.ylim([0,1])
        plt.title('Running average' + '\n' + 'S-phase(0) vs B-phase(1)' + '\n' 'N = %i, $\sigma$ = %.5f, iter = %i' %(params[1],params[7],j), fontsize=15)
        plt.xlabel('time',fontsize=15)
        plt.yticks([0,1])
        plt.xlim([window_av+window+1,t[-1]])
    return run_av[-1]

# Input parameters    
L = 25     # number of loci
hpop = pop.haploid_highd(L)
n_savings = 1001     # number of data savings (generations / sv_int)
sv_int = 1      # print data every x generations
mu = 0.5     # mutation_rate
r = 0.5     # outcrossing_rate
rho = 0.5     # crossover_rate
N = [200]
sigma = np.linspace(0.005,0.01, num=2)
n_iters = 1
rav = []
n_is_b_phase = []
for n_iter in N:
    # Inizialize clones
    spin = np.random.choice([True, False], (n_iter, L))     # MSA random 
    cln_sizes = np.random.randint(3, size=n_iter)           # Counts: the genome MSA(i,:) appears cln(i) times 
    for sigma_iter in sigma:
        # Generate F_{ij} for the simulation
        [mu_tr,sigma_tr,F] = RFM(0.0,sigma_iter, L)
        params = [L,n_iter,n_savings,sv_int,mu,r,rho,sigma_iter,mu_tr,sigma_tr]     # to pass as argument to functions !AVOID MODIFY!
        rav_iter = 0
        for j in range(n_iters):
            declare_initial(params,j)
            FFPopSim_initialization(L,n_iter,mu,r,rho,sigma_iter,spin,cln_sizes,F)
            t = []  
            chi = []
            popstat = []
            # Evolution (core)
            for s in range(n_savings):
                t.append(s*sv_int)
                hpop.unique_clones()     # group together the same clones! 
                for i in range(L): 
                    chi.append(hpop.get_chi(i))
                popstat.append([hpop.get_fitness_statistics().mean,hpop.get_fitness_statistics().variance]) 
                hpop.evolve(sv_int)
            chi = np.reshape(np.array(chi), (n_savings, L))
            ev_freqs(params,t,chi,'all')
            popstat = np.array(popstat) 
            ev_fitness_stats(t*sv_int, popstat[:,0],popstat[:,1])
            phase = classifier(t,popstat[:,0],params,j)
            rav_iter += run_average(t,phase,params)
        rav.append([sigma_iter,rav_iter/n_iters])
            
# %% End 
print("--- %f minutes ---" % ((time.time() - start_time)/60.))

""""------------------------------------------------------------
# END
"""   
    
