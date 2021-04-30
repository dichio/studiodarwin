#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 2020
Last change : Wed Mar 17 2021
@author: Vito Dichio

Modified version of crystalball.py adapted to study the distribution of escape times from QLE and NRC.
"""

# %% Import modules ##########
           
import os, sys, time, gc
import numpy as np
sys.path.append('/***/***/***/Darwin_studio') # some sub-functions 
import FFPopSim as pop
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import colors
if __name__ == "__main__": 
    print(sys.argv)

# %% Initialization & Evolution ##########
start_time = time.time()
np_rng_seed = 1 ; np.random.seed(np_rng_seed)     # Fix the ran_seed 

def RFM(mu,sigma,L): #As in cristalball.py
    """
    Random Fitnees Model: initializes an LxL F_matrix of epistatic fitness (symmetric, diagonal=0) with L(L-1)/2 gaussian numbers $\sim N(\mu,\sigma). Generated data have a sigma_true whose value has a relative error < 1/1000. (mu_true, sigma_true) in output. 
    """
    sigma_true = 10.0
    mu_true = 0.0
    while (abs(sigma_true - sigma)/sigma > 0.0001) :
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
    bwidth = 3.5
    figsizex = 20
    figsizey = 8
    fig = plt.figure(figsize=(figsizex, figsizey))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwidth)
    ax.spines['top'].set_linewidth(bwidth)
    ax.spines['left'].set_linewidth(bwidth)
    ax.spines['right'].set_linewidth(bwidth) 
    cmap = plt.get_cmap('jet_r')
    if locus == 'all':
       for loc in range(L):
           color = cmap(float(loc) / L)  # cm.cool(locus)
           plt.plot(t, chi[:, loc], ls='-', lw=1., c=color)
    else :
       color = cmap(float(locus) / L)  # cm.cool(locus)
       plt.plot(t, chi[:, locus], ls='-', lw=1., c=color)  
    char_size = 20   
    plt.xlabel('Time [generations]',fontsize=char_size)
    plt.ylabel('Allele frequencies',fontsize=char_size)
    plt.tick_params(labelsize=char_size)
    plt.tight_layout()
    #plt.xlim([15000,15250])
    #plt.ylim([,])
    plt.title('L=%i' %params[0] +' ' + 'N=%i' %params[1] +' '+ '$\mu$=%.3f' %params[4] +' '+ ' $r$=%.3f'%r +' '+ ' $\sigma_{tr}$=%.5f'%params[9], fontsize=char_size)
    #plt.show()
    plt.savefig('Plots/ev_f.png')
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
        fig = plt.figure(figsize=(10,2))
        plt.step(t[window-1:-1],phase[:], ls='-', lw=2.,color='red')
        plt.ylim([0,1])
        plt.title('QLE-phase(0) vs NRC-phase(1)' + '\n' 'N = %i, $\sigma$ = %.5f, iter = %i' %(params[1],params[7],j), fontsize=15)
        plt.xlabel('time',fontsize=15)
        plt.yticks([0,1])
        plt.xlim([0,t[-1]])
        plt.savefig('Plots/classify.png')
        #plt.show()
        plt.close(fig)
    return phase
def ev_fitness_stats(t,fit_mean,fit_var):  #As in cristalball.py
    fig = plt.figure(3,figsize=(15,5))
    plt.plot(t, fit_mean, label='fitness mean')
    plt.plot(t, np.sqrt(fit_var), label='fitness standard deviation')
    plt.legend(loc=2,fontsize=15)
    #plt.xlim([15000,15250])
    #plt.ylim([,])
    plt.xlabel('Time',fontsize=15)
    plt.title('L=%i' %params[0] +' ' + 'N=%i' %params[1] +' '+ '$\mu$=%.3f' %params[4] +' '+ ' $r$=%.3f'%r +' '+ ' $\sigma_{tr}$=%.5f'%params[9], fontsize=15)
    plt.savefig('Plots/ev_fs.png')
    #plt.show()
    plt.close(fig)
def phase_analysis(phase):
    """
    This function takes as input the array phase of 0s and 1s, detects change points in the sequence and creates two vectors: analyseS whose i-th are is (time) lenght of the i-th segment like 00...0 detected in phase i.e. QLE like behaviour; analyseB is the analogous for the NRC phase. 
    """
    ph_analysis = []
    window = 1
    st_point = window
    for i in range(window,len(phase)):
        if phase[i]!=phase[i-1]:
            ph_analysis.append([i-st_point,phase[i-1]])
            st_point = i                 
    analyzeB = []
    analyzeS = []
    if len(ph_analysis) > 1:
        for i in range(1,len(ph_analysis)): # discard the first one!
            if ph_analysis[i][1] == 0:
                analyzeS.append(ph_analysis[i][0])
            elif ph_analysis[i][1] == 1:
                analyzeB.append(ph_analysis[i][0])
            else : 
                print('Something\'s wrong in the phase analysis!')
    #print(ph_analysis)
    return analyzeS,analyzeB
# Input parameters    
L = 25     # number of loci
hpop = pop.haploid_highd(L)
n_savings = 150001     # number of data savings (generations / sv_int)
sv_int = 1      # print data every x generations
mu = 0.5     # mutation_rate
r = 0.5     # outcrossing_rate
rho = 0.5     # crossover_rate
#N = np.linspace(640,700,num=1,dtype=int)
N = [725]
sigma = np.linspace(0.0291,0.0291, num=1)

n_iters = 10
analyzeS = []
analyzeB = []

#%%
n_is_b_phase = []
for sigma_iter in sigma:
    # Generate F_{ij} for the simulation
    [mu_tr,sigma_tr,F] = RFM(0.0,sigma_iter, L)
    for n_iter in N:
        # Inizialize clones
        spin = np.random.choice([True, False], (n_iter, L))     # MSA random 
        cln_sizes = np.random.randint(3, size=n_iter)     # Counts: the genome MSA(i,:) appears cln(i) times 
        params = [L,n_iter,n_savings,sv_int,mu,r,rho,sigma_iter,mu_tr,sigma_tr]     # to pass as argument to functions !AVOID MODIFY!
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
            tempS,tempB = phase_analysis(phase)
            analyzeB.append(tempB) 
            analyzeS.append(tempS)
analyzeS = sum(analyzeS,[]) #merge in a unique list
analyzeB = sum(analyzeB,[])
#%% Info on the fitness values used
printF = False
if printF :
    print(F)
    plt.hist(np.reshape(F,L*L),50)
    plt.ylabel('counts', fontsize=15)
    plt.xlabel('$f_{ij}$', fontsize=15)
    plt.title('Quenched distribution of $f_{ij}\sim\mathcal{N}(0,\sigma)$, $\sigma$ = %.5f' %sigma_tr, fontsize=15)

    with open('fit_land.dat', 'w') as filehandleF:
      for line in np.array(F):
         np.savetxt(filehandleF, line, fmt='%.5f')
#%%
fromfile = False 
printfile = False
logscale = False

if fromfile:
    N = [750]
    params = [L,N[0],n_savings,sv_int,mu,r,rho,0,0,sigma[0]]
    analyzeS = np.loadtxt('/***/***.dat' %N[0]) #where to find analyseB?
    analyzeB = np.loadtxt('/h***/***.dat' %N[0])

n_bins = 50
def cm(x): 
    return x*0.3937

def f(x, A, B):
    return A*x + B
def histfit(ticks,x,color,phase): #Linear fit in semi-log scale, evaluation of parameter "a" in y=c*Exp[a*x]
    loghist = np.array([[ticks[i],np.log(x[i])] for i in range(len(x)) if x[i]>1])
    popt, pcov = curve_fit(f, loghist[:,0],loghist[:,1]) # your data x, y to fit
    print('y = c*Exp[a*x]' + '\n' + '****************' + '\n' + '* a = %.5f *' %popt[0]+ '\n' + '****************')
    print(popt)
    print(pcov)
    x = np.linspace(1,ticks[-1],num=500)
    plt.plot(x,np.e**popt[1]*np.exp(x*popt[0]),color,label=('Exp fit ' + phase + ' phase'),lw=.8)
    return popt[0]    

def plot_escape(analyzeS,analyzeB):    
    fig, ax = plt.subplots(figsize=(cm(8),cm(8)))
    char_size = 7
    label_size = 6
    plt.hist(analyzeS,n_bins,label="QLE - phase",color = 'gold',zorder=1)
    histogS,edges = np.histogram(analyzeS,n_bins) 
    ticksS = [edges[i-1]+(edges[i]-edges[i-1])/2. for i in range(1,len(edges))]
    plt.plot(ticksS,histogS,'black',linestyle ='-.',label='_nolegend_',lw=.3)
    expS = histfit(ticksS,histogS,'black','QLE')
    
    plt.hist(analyzeB,n_bins,label='NRC - phase', color = 'gray',zorder=0)
    histogB,edges = np.histogram(analyzeB,n_bins) 
    ticksB = [edges[i-1]+(edges[i]-edges[i-1])/2. for i in range(1,len(edges))]
    plt.plot(ticksB,histogB,'red',linestyle='-.',label='_nolegend_',lw=.3)
    expB = histfit(ticksB,histogB,'red','NRC')

    plt.tick_params(labelsize=label_size)    
    ax.ticklabel_format(style='scientific',axis='x',scilimits=(0,0))
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(label_size)
    plt.ylabel('counts', fontsize=char_size)
    plt.ylim(bottom = 1/np.e,top = max(max(histogS),max(histogB)))
    plt.xlabel('Escape Time', fontsize=char_size)
    plt.xlim(left=1,right=max(ticksS[-1],ticksB[-1]))
    plt.title('Distribution of persistence in NRC/QLE phase' + '\n' + 'L=%i' %params[0] +' ' + 'N=%i' %params[1] +' '+ '$\mu$=%.3f' %params[4] +' '+ ' $r$=%.3f'%r +' '+ ' $\sigma_{tr}$=%.5f'%params[9] + ' ' +'T = %i' %n_savings + '\n' + 'iters = %i' %n_iters + '\n' + 'y_{fit} = c*Exp[a*x]' + '\n' + 'a_S = %.5f' %expS + '\n' + 'a_B = %.5f' %expB, fontsize=char_size)
    plt.title('N = %i' %N[0], fontsize=char_size)
    plt.legend(loc='best',fontsize=label_size)
    #plt.xlim([0,2000])
    plt.grid(lw=.3)

    if logscale:
        plt.yscale('log',basey=np.e)
        import matplotlib.ticker as mtick
        def ticks(y, pos):
            return r'$e^{:.0f}$'.format(np.log(y))
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(ticks))
    
    #plt.show()
    plt.savefig('Plots/analysis.png')
    if printfile :
        with open('S.s.%.5f-%i.dat' %(sigma[0],N[0]), 'w') as filehandleS:
            filehandleS.writelines("%i\n" %x for x in analyzeS)
        with open('B.s.%.5f-%i.dat' %(sigma[0],N[0]), 'w') as filehandleB:
            filehandleB.writelines("%i\n" %x for x in analyzeB)
        
plot_escape(analyzeS,analyzeB)    
# %% End 
print("--- %f minutes ---" % ((time.time() - start_time)/60.))

""""------------------------------------------------------------
# END
"""   
    
    
