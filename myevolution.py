#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last modified: Fri Mar 24 20:04:42 2023

Note: this code could be written in a M-U-C-H more organized way, I swear I can do better than this. If you want to complain or you have questions, here my email: dichio.vito@gmail.com

@author: vito.dichio
"""

# %% Import modules ##########
           
import os, sys, time, gc
# Specify location of the main folder
sys.path.append('your/path')

import numpy as np
import FFPopSim as pop

import matplotlib.pyplot as plt
from matplotlib import colors
if __name__ == "__main__": 
    print(sys.argv)

# %% Initialization & Evolution ##########

start_time = time.time()

# Input parameters    
N = 500     # carrying_capacity
L = 25     # number of loci
n_savings = 10000    # number of data savings (generations / sv_int)
sv_int = 1     # print data every x generations
mu = .5     # mutation_rate
r = .5     # outcrossing_rate
rho = .5     # crossover_rate
sigma = 0.024     # st_dev gaussian generation f_{ij}

cols = ["#393E46","#6D9886","#F2E7D5","#F7F7F7"]
cols1 = ["#1572A1","#A9333A","#FBF3E4","#DFD8CA"]
cols2 = ["000000","#541212","#8B9A46","#EEEEEE"]
         
 #%%        
# Inizialize clones
np_rng_seed = 1 ; np.random.seed(np_rng_seed)      # Fix the ran_seed 
spin = np.random.choice([True, False], (N, L))     # MSA random 
cln_sizes = np.random.randint(3, size=N)           # Counts: the genome MSA(i,:) appears cln(i) times 
#cln_sizes = np.full(N,1)                          # Alternatively, all single-copies

# Generate F_{ij} for the simulation
def RFM(mu,sigma,L):
    """
    Random Fitnees Model: initializes an LxL F_matrix of epistatic fitness (symmetric, diagonal=0) with L(L-1)/2 gaussian numbers $\sim N(\mu,\sigma)$. Generated data have a sigma_true whose value has a relative error < 1/1000. 
    """
    sigma_true = 10.0
    mu_true = 0.0
    while (abs(sigma_true - sigma)/sigma > 0.001) :
        Fij = np.random.normal(mu, sigma, size=int(L*(L-1)/2)) 
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

[mu_tr,sigma_tr,F] = RFM(0.0,sigma,L)

params = [L,N,n_savings,sv_int,mu,r,rho,sigma,mu_tr,sigma_tr]     # to pass as argument to functions !DO NOT MODIFY!
#        [0 1     2        3   4  5  6   7      8      9    ] 
hpop = pop.haploid_highd(L)

def declare_initial(params):
    """
    Print of initial parameters of the system.
    """
    print('*--*--*--*--*--*--*--*--*')
    print('|')
    print('*  L = %d' %params[0])
    print('|  N = %d' %params[1])
    print('*  T = %d' %(params[2]*params[3]))
    print('|  mu = %.5f' %params[4])
    print('*  r = %.5f' %params[5])
    print('|  rho = %.5f' %params[6])
    print('*  sigma = %.5f' %params[7])
    print('|  (4*mu+1/2*r)/sigma = %f' %((4*params[4]+params[5]*1./2)/params[7]))
    print('*  True (mean, st_d) = (%.5f, %.5f)' %(params[8],params[9])) 
    print('|')
    print('*--*--*--*--*--*--*--*--*')
    print('')
declare_initial(params)

def FFPopSim_initialization(L,N,mu,r,rho,sigma,spin,cln_sizes,F):
    """
    FFPopSim is initialized with input parameters
    """
    hpop.recombination_model = pop.CROSSOVERS # FREE_RECOMBINATION/SINGLE_CROSSOVER/CROSSOVER
    hpop.carrying_capacity = N # expected size of the population
    hpop.outcrossing_rate = r # probability of sexual reproduction per generation
    hpop.crossover_rate = rho # crossover rate per site per generation
    hpop.mutation_rate = mu
    hpop.circular = False # circular genotype
    hpop.set_genotypes(spin, cln_sizes) # Input genotypes
    hpop.clear_fitness() # add pairwise fitness coefficients to fitness function
    for i in range(L):
        for j in range(L):
            hpop.add_fitness_coefficient(F[i][j], [i, j])

FFPopSim_initialization(L,N,mu,r,rho,sigma,spin,cln_sizes,F)

def cm(x): 
    # From cm to goddam inches 
    return x*0.3937

# Some plots are defined to observe ongoing evolution. 
char_size = 8
label_size = 8
def control_plots(params,s,choose_plots):
    '''
    Control plots: istantaneous observables during pop. evolution. hpop = FFPopSim.haploid_highd(L), params is the list of parameters, s is the generation number, choose_plots=[-,-,-,-] is an array of 4 logical variables: 0-pop.snapshot 1-fitness_hist, 2-overlap_distrib, 3-divergence_hist.
    '''
    T = params[2]*params[3] - 1
    if choose_plots[0]==True:
        pop_snapshot(params[1],s,T) 
    if choose_plots[1]==True:
        pop_fitness_hist(params[1],s,T,3,5)      
def pop_snapshot(N,s,T): #fancy
    fig = plt.figure()
    N_samples = 200
    if N < N_samples :
        print('More grid plot samples than the carrying capacity!')
    plt.figure(dpi=200,figsize=(cm(17),cm(6)))
    plt.title('t = %i' %s, fontsize=char_size)
    cmap = colors.ListedColormap([cols[index] for index in [0,-1]])
    plt.imshow(hpop.random_genomes(N_samples).T,cmap=cmap)  
    plt.tick_params(labelsize=label_size)
    plt.yticks([])
    plt.xticks([])
    plt.ylabel('$s_i$',fontsize=char_size)
    plt.xlabel(' %i samples' %(N_samples),fontsize=char_size)
    plt.savefig('Plots/sns%i.png' %s, dpi=200, bbox_inches = "tight" )
    plt.close(fig)          

def pop_fitness_hist(N,s,T,x1,x2) :
    fig = plt.figure(figsize=(cm(15),cm(6)))
    ax = plt.gca()
    hpop.plot_fitness_histogram(ax,n_sample=N,bins=20,density=False,color=cols[1],edgecolor=cols[1],linewidth=1.5)
    plt.xlabel('Fitness', fontsize=char_size)
    plt.title('t =  %i' %s, fontsize=char_size)
    plt.tick_params(labelsize=label_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim([-2,4.5])
    
    a = plt.axes([.65, .7, .25, .2], facecolor='w')        # Zoom in a specific region of the histogram as subplot 
    hpop.plot_fitness_histogram(a,n_sample=N,bins=100,density=False,color = cols[0],linewidth=0)
    plt.tick_params(labelsize=label_size,rotation=0)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_visible(False)
    a.yaxis.tick_right()
    plt.xlim([x1,x2])
    #plt.title('zoom', fontsize=char_size)
    plt.savefig('Plots/fd_T%i.pdf' %s, dpi=600, bbox_inches = "tight" )
    plt.show()
    plt.close(fig)

print('Let the population evolve!')

calibration_time = 0
if calibration_time != 0 :
    hpop.evolve(calibration_time)     # Intial calibration, if needed!

# Initialize empty storages
t = []     # n_savings - list of times of data savings   
chi = []     # n_savings x L - list of single locus magnetization
chi2 = []     # n_savings x L^2 - list of pair correlations 
popstat = []     # n_savings x n_statistics - list of 4 pop. statistics : fitness mean, fitness variance , participation ratio, number of clones
clone_sizes = []     # n_savings x n_clones(t) - list of clone sizes (differenent lenghts!)

# Evolution (core)
for s in range(n_savings):
    if (s*sv_int) % 200 == 0 or s == n_savings-1: # Plot the obsevables every Y=500 generations (your choice!)
        if (s*sv_int) >= 1: 
            control_plots(params,s,[True,True])
    hpop.unique_clones()     # group together clones with the same genotype!
    t.append(s*sv_int)
    if s == 0: # n_savings x (N*L) - list of pop. genotypes
        genotypes_records =  hpop.get_genotypes()
    else :    
        genotypes_records = np.vstack((genotypes_records, hpop.get_genotypes()))
    for i in range(L): 
        chi.append(hpop.get_chi(i))
        for j in range(L):
            chi2.append(hpop.get_chi2(i, j))
    popstat.append([hpop.get_fitness_statistics().mean,hpop.get_fitness_statistics().variance, hpop.participation_ratio, hpop.number_of_clones])        
    clone_sizes.append(sorted( [x for (i,x) in enumerate(hpop.get_clone_sizes())], reverse=True ))
    hpop.evolve(sv_int)
print('End of evolution!') 

# %% Convert some lists into arrays 
chi = np.reshape(np.array(chi), (n_savings, L))
chi2 = np.reshape(np.array(chi2), (n_savings, L * L))
popstat = np.array(popstat)

np.save('output/chi.npy', chi) 
np.save('output/chi2.npy', chi2) 
np.save('output/ps.npy', popstat) 

# %% All-Time Plots 
# Plots to observe population statistics as function of time, namely clonal structure of the population, number of clones, participation ratio, fitness mean and st.dev., frequencies of loci, correlations between loci
char_size = 8
label_size = 8
n=10
figsizex = cm(17)
figsizey = cm(4)
mydpi = 600

def smoother(x,dt=100):
    D = np.shape(x)[0]
    chi_smooth = []
    for i in range(dt,D):
        chi_smooth.append(np.mean(x[i-dt:i]))
    return(np.array(chi_smooth))
    
#%%
def ev_clonesizes(t,cl_sizes): #fancy
    """
    Evolution of clone structure of the population. cl_sizes (list of lists with different lenghts) is copied in a uniform matrix (gaps filled with zeroes)
    """
    lattice = np.zeros([len(cl_sizes),len(max(cl_sizes,key = lambda x: len(x)))])
    for i,j in enumerate(cl_sizes):
        lattice[i][0:len(j)] = j
    for i in t:
        lattice[i][:] = lattice[i][:]/np.sum(lattice,axis=1)[i] 
    lattice = np.transpose(np.array(lattice))      
    fig, ax = plt.subplots(figsize=(cm(9),cm(6)))
    pal = ['#fef200','#000000','#00adef','#FF0000','#ffffff','#0000ff']
    ax.stackplot(t,lattice,colors=pal,lw=1)
    ax.ticklabel_format(style='scientific',axis='x',scilimits=(0,0))
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(label_size)
    plt.xlabel('t',fontsize=char_size)
    plt.ylim([0,1.0])
    plt.xlim([0,params[2]])
    plt.xlim([0,100])
    plt.tick_params(labelsize=label_size)
    #plt.savefig('Plots/ev_cs.png')
    #char_size_red = 6
    #a = plt.axes([.5, .5, .35, .35], facecolor='y')  #Zoom in a region of interest as subplot   
    #pal = ['#fef200','#000000','#00adef','#FF0000','#ffffff','#0000ff']
    #a.stackplot(t,lattice,colors=pal,lw=1)
    #a.ticklabel_format(style='scientific',axis='x',scilimits=(0,0))
    #tx = a.xaxis.get_offset_text()
    #tx.set_fontsize(char_size_red)
    #plt.tick_params(labelsize=char_size_red)
    #plt.ylim([0,0.1])
    #plt.xlim([0,params[2]])
    plt.show()
    plt.close(fig)

#%%
def ev_fitness_stats(t,fit_mean,fit_var): #fancy
    fig = plt.figure(figsize=(figsizex, figsizey))
    ax = plt.gca()
    plt.plot(t[::n], fit_mean[::n], lw=1.0, label='mean', color = cols[1])
    #plt.plot(t, np.sqrt(fit_var), lw=0.2, label='fitness standard deviation')
    plt.fill_between(t[::n], fit_mean[::n]-np.sqrt(fit_var)[::n],fit_mean[::n]+np.sqrt(fit_var)[::n],alpha=.2, edgecolor=cols[1], facecolor=cols[1],
    linewidth=.2, linestyle='-',label='st.dev.')#, antialiased=True)
    plt.legend(loc=1,fontsize=char_size,ncol=2,handlelength=1)
    plt.xlim([0,params[2]])
    #plt.ylim([-2,4])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(labelsize=char_size,rotation=0)
    #ax.ticklabel_format(style='scientific',axis='x',scilimits=(0,0))
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(label_size)
    plt.xlabel('t',fontsize=char_size)
    plt.tick_params(labelsize=label_size)
    #plt.title('L=%i' %params[0] +' ' + 'N=%i' %params[1] +' '+ '$\mu$=%.2f' %params[4] +' '+ ' $r$=%.2f'%r +' '+ ' $\sigma_{tr}$=%.3f'%params[9], fontsize=char_size)
    plt.savefig('Plots/ev_fs.pdf',dpi=mydpi,bbox_inches = "tight")
    plt.show()
    plt.close(fig) 

def ev_fitness_stats_zoom(t,fit_mean,fit_var): #fancy
    fig = plt.figure(figsize=(cm(12), figsizey))
    ax = plt.gca()
    plt.plot(t[::n], fit_mean[::n], lw=1.0, label='mean fitness', color = cols[1])
    plt.xlim([0,params[2]])
    #plt.ylim([-2,4])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(labelsize=char_size,rotation=0)
    #ax.ticklabel_format(style='scientific',axis='x',scilimits=(0,0))
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(label_size)
    plt.xlabel('t',fontsize=char_size)
    plt.ylabel('',fontsize=char_size)
    plt.tick_params(labelsize=label_size)
    #plt.title('L=%i' %params[0] +' ' + 'N=%i' %params[1] +' '+ '$\mu$=%.2f' %params[4] +' '+ ' $r$=%.2f'%r +' '+ ' $\sigma_{tr}$=%.3f'%params[9], fontsize=char_size)
    plt.savefig('Plots/ev_fs.pdf',dpi=mydpi,bbox_inches = "tight")
    plt.show()
    plt.close(fig) 
    
ev_fitness_stats_zoom(t*sv_int, popstat[:,0],popstat[:,1])

#%%      
def ev_freqs(params,t,chi,locus): 
    bwidth = 0.5
    fig = plt.figure(figsize=(figsizex, figsizey))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwidth)
    ax.spines['top'].set_linewidth(bwidth)
    ax.spines['left'].set_linewidth(bwidth)
    ax.spines['right'].set_linewidth(bwidth) 
    
    dt = 10
    cl = 0
    for lc in locus:
        print(cl)
        plt.plot(t[::n], chi[::n, lc], ls='-', lw=1, c=cols1[cl], alpha=0.5,label='$\chi_{%i}$' %(lc+1))    
        plt.plot(t[dt:], smoother(chi[:,lc],dt), ls='-', lw=.5, c=cols1[cl])
        cl +=1
    
    plt.legend(loc=0,fontsize=char_size,ncol=2,handlelength=2)
    plt.xlabel('t',fontsize=char_size)
    plt.xlim([0,params[2]])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(labelsize=label_size)
    #ax.ticklabel_format(style='scientific',axis='x',scilimits=(0,0))
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(label_size)
    #plt.title('L=%i' %params[0] +' ' + 'N=%i' %params[1] +' '+ '$\mu$=%.2f' %params[4] +' '+ ' $r$=%.2f'%r +' '+ ' $\sigma_{tr}$=%.3f'%params[9], fontsize=char_size)
    plt.savefig('Plots/ev_f.pdf',dpi=mydpi,bbox_inches = "tight")
    plt.show()
    plt.close()
    
ev_freqs(params,t,chi,[0,2])


#%%
def ev_corr(params,t,chi2,locus): 
    bwidth = 0.5
    fig = plt.figure(figsize=(figsizex, figsizey))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwidth)
    ax.spines['top'].set_linewidth(bwidth)
    ax.spines['left'].set_linewidth(bwidth)
    ax.spines['right'].set_linewidth(bwidth) 
    
    dt = 50
    cl = 0
    #plt.plot(t[::n], 1-chi2[::n, 0], ls='-', lw=.05, c=cols2[cl])
    #plt.plot(t[dt:], 1-smoother(chi2[:,0],dt), ls='-', lw=.5, c=cols2[cl], label='$1-\chi_{0%i}$' %0)
    for lc in locus:
        cl += 1
        #plt.plot(t[::n], chi2[::n, lc], ls='-', lw=.05, c=cols2[cl])  
        plt.plot(t[dt:], smoother(chi2[:,lc],dt), ls='-', lw=.5, c=cols2[cl], label='$\chi_{1%i}$' %(lc+1))
    
    plt.legend(loc=3,fontsize=char_size,ncol=3,handlelength=2)
    plt.xlabel('t',fontsize=char_size)
#    plt.ylabel('$\{\chi_{ij}(t)\}_{j=1}^L$',fontsize=char_size)   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(labelsize=label_size)
    #ax.ticklabel_format(style='scientific',axis='x',scilimits=(0,0))
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(label_size)
    plt.xlim([0,params[2]])
    plt.ylim([-0.1,0.1])
    #plt.title('L=%i' %params[0] +' ' + 'N=%i' %params[1] +' '+ '$\mu$=%.2f' %params[4] +' '+ ' $r$=%.2f'%r +' '+ ' $\sigma_{tr}$=%.3f'%params[9], fontsize=char_size)
    plt.savefig('Plots/ev_c.pdf',dpi=mydpi,bbox_inches = "tight")
    plt.show()
    plt.close()

ev_corr(params,t,chi2,[1,2])

# %% End
print("--- %f minutes ---" % ((time.time() - start_time)/60.))

""""------------------------------------------------------------
# END
"""
    
