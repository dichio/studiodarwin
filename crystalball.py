#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 2020
Edited for GitHub: Wed Apr 28 2021
@author: Vito Dichio
"""

# %% Import modules ##########
           
import os, sys, time, gc
import numpy as np
sys.path.append('/Users/vito.dichio/Desktop/studiodarwin') # Specify location of the main folder
import inference #some subfunction for inference purposes
import FFPopSim as pop
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
if __name__ == "__main__": 
    print(sys.argv)

# %% Initialization & Evolution ##########

start_time = time.time()

# Input parameters    
N = 200     # carrying_capacity
L = 25     # number of loci
n_savings = 2501     # number of data savings (generations / sv_int)
sv_int = 1      # print data every x generations
mu = .05     # mutation_rate
r = .5     # outcrossing_rate
rho = .5     # crossover_rate
sigma = 0.002     # st_dev gaussian generation f_{ij}

# Inizialize clones
np_rng_seed = 1 ; np.random.seed(np_rng_seed)      # Fix the ran_seed 
spin = np.random.choice([True, False], (N, L))     # MSA random 
#spin = np.ones((N, L), dtype=bool)                # Alternatively, MSA all -1
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
def RFM_as(mu,sigma,L):
    """
    Random Fitnees Model: initializes an LxL F_matrix of epistatic fitness (asymmetric, diagonal=0) with L(L-1)/2 gaussian numbers $\sim N(\mu,\sigma). Generated data have a sigma_true whose value has a relative error < 1/1000. 
    """
    sigma_true = 10.0
    mu_true = 0.0
    while (abs(sigma_true - sigma)/sigma > 0.001) :
        Fij = np.random.normal(mu, sigma, size=(L*(L-1))) 
        mu_true = np.mean(Fij)
        sigma_true = np.sqrt(np.var(Fij))
    F_matrix = np.zeros((L,L))
    k = 0
    for i in range(0,L):
        for j in range(0,L):
            if i != j :
                F_matrix[i][j] = Fij[k]
                k += 1 
    return [mu_true, sigma_true, F_matrix]

[mu_tr,sigma_tr,F] = RFM(0.0,sigma,L)

params = [L,N,n_savings,sv_int,mu,r,rho,sigma,mu_tr,sigma_tr]     # to pass as argument to functions !AVOID MODIFY!
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
char_size = 7
label_size = 6
def control_plots(params,s,choose_plots):
    '''
    Control plots: istantaneous observables during pop. evolution. hpop = FFPopSim.haploid_highd(L), params is the list of parameters, s is the generation number, choose_plots=[-,-,-,-] is an array of 4 logical variables: 0-pop.snapshot 1-fitness_hist, 2-overlap_distrib, 3-divergence_hist.
    '''
    T = params[2]*params[3] - 1
    if choose_plots[0]==True:
        pop_snapshot(params[1],s,T) 
    if choose_plots[1]==True:
        pop_fitness_hist(params[1],s,T,3,4.75)
    if choose_plots[2]==True:
        pop_overlap_hist(params[1],s,T)
    if choose_plots[3]==True:
        pop_divergence_hist(params[1],s,T)        
def pop_snapshot(N,s,T): #fancy
    fig = plt.figure()
    N_samples = 200
    if N < N_samples :
        print('More grid plot samples than the carrying capacity!')
    plt.figure(dpi=200,figsize=(cm(10),cm(3)))
    plt.title('t = %i' %s, fontsize=char_size)
    cmap = colors.ListedColormap(['black', 'yellow'])
    plt.imshow(hpop.random_genomes(N_samples).T,cmap=cmap)  
    plt.tick_params(labelsize=label_size)
    plt.yticks([])
    plt.xticks([])
    plt.ylabel('$\{s_i\}_{i=1}^L$',fontsize=char_size)
    plt.xlabel(' - %i - random samples out of %i' %(N_samples,N),fontsize=char_size)
    plt.close(fig)
def pop_divergence_hist(N,s,T):
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    hpop.plot_divergence_histogram(n_sample=N)
    plt.title('T =  - %i - out of %i' %(s,T), fontsize=15)
    plt.xlabel('Divergence',fontsize=15)
    plt.close(fig)          
def pop_overlap_hist(N,s,T):
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    hpop.plot_diversity_histogram(n_sample=N)
    plt.title('t =  %i' %s, fontsize=char_size)
    plt.xlabel('Diversity',fontsize=char_size)
    plt.tick_params(labelsize=char_size,rotation=0)
    #plt.savefig('Plots/ol_T%i.png' %s )
    plt.close(fig)
def pop_fitness_hist(N,s,T,x1,x2) :
    fig = plt.figure(figsize=(cm(9),cm(6)))
    ax = plt.gca()
    hpop.plot_fitness_histogram(ax,n_sample=N)
    plt.xlabel('Fitness', fontsize=char_size)
    plt.title('t =  %i' %s, fontsize=char_size)
    plt.savefig('Plots/fh_T%i.png' %s, dpi=200, bbox_inches = "tight")
    plt.tick_params(labelsize=label_size)
    #a = plt.axes([.625, .6, .25, .2], facecolor='y')        # Zoom in a specific region of the histogram as subplot 
    #hpop.plot_fitness_histogram(a,n_sample=N,bins=100)
    #plt.tick_params(labelsize=label_size,rotation=0)
    #plt.xlim([x1,x2])
    #plt.title('zoom', fontsize=char_size)
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
    if (s*sv_int) % 500 == 0 or s == n_savings-1: # Plot the obsevables every Y=500 generations (your choice!)
        if (s*sv_int) >= 1: 
            control_plots(params,s,[False,False,False,False])
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

# Convert some lists into arrays 
chi = np.reshape(np.array(chi), (n_savings, L))
chi2 = np.reshape(np.array(chi2), (n_savings, L * L))
popstat = np.array(popstat)


# %% All-Time Plots 
# Plots to observe population statistics as function of time, namely clonal structure of the population, number of clones, participation ratio, fitness mean and st.dev., frequencies of loci, correlations between loci
char_size = 8
label_size = 7
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
def ev_num_clones(t,n_clones):
    fig = plt.figure(2,figsize=(10,5))
    plt.plot(t, n_clones)
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Number of clones',fontsize=15)
    #plt.xlim([0,500])
    #plt.ylim([1,300])
    #plt.xscale('log')
    #plt.yscale('log')
    plt.title('L=%i' %params[0] +' ' + 'N=%i' %params[1] +' '+ '$\mu$=%.3f' %params[4] +' '+ ' $r$=%.3f'%r +' '+ ' $\sigma_{tr}$=%.5f'%params[9], fontsize=15)
    #plt.savefig('Plots/ev_nc.png')
    plt.show()
    plt.close(fig)
def ev_part_ratio(t,part_ratio):
    fig = plt.figure(5,figsize=(15,5))
    plt.plot(t, part_ratio)
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Paticipation ration',fontsize=15)
    #plt.xlim([,])
    #plt.ylim([,])
    #plt.savefig('Plots/ev_pr.png')
    plt.show()
    plt.close(fig)
def ev_fitness_stats(t,fit_mean,fit_var): #fancy
    fig = plt.figure(figsize=(cm(12),cm(5)))
    ax = plt.gca()
    plt.plot(t, fit_mean, lw=.15, label='fitness mean')
    plt.plot(t, np.sqrt(fit_var), lw=0.2, label='fitness standard deviation')
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
    #plt.savefig('Plots/ev_fs.png',dpi=200,bbox_inches = "tight")
    plt.show()
    plt.close(fig)       
def ev_freqs(params,t,chi,locus): #fancy
    """
    Evolution of allele frequencies. Specify locus. If all_loci then set locus = 'all'.
    """
    if locus >= L and not locus == 'all': 
        print('This locus does not exist!')
    bwidth = 0.5
    figsizex = cm(12)
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
    plt.savefig('Plots/ev_f.png',dpi=200,bbox_inches = "tight")
    fig.tight_layout()
    plt.close()
def ev_corr(params,t,chi2,locus): #fancy
    """
    Evolution of allele correlations. Specify one locus \chi_i to see all \chi_ij, j=0,L-1.
    """
    if locus >= L :
        print('This locus does not exist!')
    bwidth = 0.5
    figsizex = cm(12)
    figsizey = cm(6)
    fig = plt.figure(figsize=(figsizex, figsizey))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwidth)
    ax.spines['top'].set_linewidth(bwidth)
    ax.spines['left'].set_linewidth(bwidth)
    ax.spines['right'].set_linewidth(bwidth) 
    cmap = plt.get_cmap('prism_r')
    start = locus*L 
    #cmap = plt.get_cmap('jet')
    for locuslocus in range(start,start+L):
        color = cmap((float(locuslocus-start) / L))  # cm.cool(locus)
        plt.plot(t, chi2[:, locuslocus], ls='-', lw=0.15, c=color)  
    plt.xlabel('t',fontsize=char_size)
#    plt.ylabel('$\{\chi_{ij}(t)\}_{j=1}^L$',fontsize=char_size)   
    ax.text(.5,1.05,'$\{\chi_{ij}(t)\}_{j=1}^L$', horizontalalignment='center', transform=ax.transAxes,fontsize=char_size) 
    plt.tick_params(labelsize=label_size)
    ax.ticklabel_format(style='scientific',axis='x',scilimits=(0,0))
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(label_size)
    plt.tight_layout()
    plt.xlim([0,params[2]])
    #plt.ylim([,])
    #plt.title('L=%i' %params[0] +' ' + 'N=%i' %params[1] +' '+ '$\mu$=%.2f' %params[4] +' '+ ' $r$=%.2f'%r +' '+ ' $\sigma_{tr}$=%.3f'%params[9], fontsize=char_size)
    plt.savefig('Plots/ev_c.png',dpi=500,bbox_inches = "tight")
    plt.close()

# Choose the all-time plots you prefer!

#ev_clonesizes(t,clone_sizes)
#ev_num_clones(t*sv_int, popstat[:,-1])
#ev_part_ratio(t*sv_int, popstat[:,-2])
#ev_fitness_stats(t*sv_int, popstat[:,0],popstat[:,1])
ev_freqs(params,t,chi,'all')
ev_corr(params,t,chi2,1)

# %% Inference
shall_I_infer = True
if shall_I_infer :
    print('Time to infer!')
    AT_chi = np.mean(chi, axis=0)     # axis=0 means along columns!
    AT_chi2 = np.reshape(np.mean(chi2, axis=0), (L, L))
    #print(AT_chi)
    rc, inveps = inference.c_ij_eval(params)
    inveps = (4*mu+0.5*r)*np.ones((L,L))
    inveps_Mon = (4*mu+0.5*(-np.log(1-r)))*np.ones((L,L))

    techs = [False,True,True]     # 0 -> inf. from correlations , 1 -> nMF, 2 -> PLM
    formulas = [True,False,False,False]     # 0 -> NS, 1 -> GC I order, 2 -> GC II order
    
    if techs[2] == True:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        J_plm = np.array(eng.plm_for_FFpopsim(matlab.logical(genotypes_records.tolist())))
        if formulas[0] == True:
            f_plm_NS = inference.infer_formulas(rc,J_plm,'I')
            e_plm_NS = inference.err_epsilon(f_plm_NS,F)
        if formulas[1] == True:
            f_plm_GCI = inference.infer_formulas(inveps,J_plm,'I')
            e_plm_GCI = inference.err_epsilon(f_plm_GCI,F)
        if formulas[2] == True:
            f_plm_GCII = inference.infer_formulas(inveps,J_plm,'II')
            e_plm_GCII = inference.err_epsilon(f_plm_GCII,F)
        if formulas[3] == True:
            f_plm_GC_impr = inference.infer_improved(inveps,J_plm,AT_chi)
            e_plm_GC_impr = inference.err_epsilon(f_plm_GC_impr,F)
    if techs[0] == True:
        if formulas[0] == True:
            f_corr_NS = inference.infer_formulas(rc,AT_chi2,'I')
            e_corr_NS = inference.err_epsilon(f_corr_NS, F)
        if formulas[1] == True:
            f_corr_GCI = inference.infer_formulas(inveps,AT_chi2,'I')
            e_corr_GCI = inference.err_epsilon(f_corr_GCI, F)
            f_corr_GCI_Mon = inference.infer_formulas(inveps_Mon,AT_chi2,'I')
            e_corr_GCI_Mon = inference.err_epsilon(f_corr_GCI_Mon, F)
        if formulas[2] == True:
            f_corr_GCII = inference.infer_formulas(inveps,AT_chi2,'II')
            e_corr_GCII = inference.err_epsilon(f_corr_GCII, F)
        if formulas[3] == True:
            f_corr_GC_impr = inference.infer_improved(inveps,AT_chi2,AT_chi)
            e_corr_GC_impr = inference.err_epsilon(f_corr_GC_impr,F)
            f_corr_GC_impr_Mon = inference.infer_improved(inveps_Mon,AT_chi2,AT_chi)
            e_corr_GC_impr_Mon = inference.err_epsilon(f_corr_GC_impr_Mon,F)
    if techs[1] == True:
        J_nMF = inference.nMF_Fij(AT_chi2)
        if formulas[0] == True:
            f_nMF_NS = inference.infer_formulas(rc,J_nMF,'I')
            e_nMF_NS = inference.err_epsilon(f_nMF_NS, F)
        if formulas[1] == True:
            f_nMF_GCI = inference.infer_formulas(inveps,J_nMF,'I')
            e_nMF_GCI = inference.err_epsilon(f_nMF_GCI, F)
        if formulas[2] == True:
            f_nMF_GCII = inference.infer_formulas(inveps,J_nMF,'II')
            e_nMF_GCII = inference.err_epsilon(f_nMF_GCII, F)
        if formulas[3] == True:
            f_nMF_GC_impr = inference.infer_improved(inveps,J_nMF,AT_chi)
            e_nMF_GC_impr = inference.err_epsilon(f_nMF_GC_impr,F)
      
#%% Inference Output
shall_I_plot = True
if shall_I_infer and shall_I_plot:
    figlabelsize = 15
    bwidth = 3.5
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwidth)
    ax.spines['top'].set_linewidth(bwidth)
    ax.spines['left'].set_linewidth(bwidth)
    ax.spines['right'].set_linewidth(bwidth)
    plt.tick_params(labelsize=figlabelsize)
    
    plt.scatter(F, f_plm_NS, alpha=0.6, color='red', label='PLM; eps=%.3f' % e_plm_NS,zorder=1)
    plt.scatter(F, f_nMF_NS, alpha=0.6, color='green', label='nMF; eps=%.3f' % e_nMF_NS,zorder=2)

    lim = sigma_tr * 5
    x = [-lim, lim]
    y = [-lim, lim]
    plt.plot(x, y, '-')
    plt.ylim([-lim, lim])
    plt.xlim([-lim, lim])
    plt.title('N = %i '%params[1] + 'L = %i ' %params[0] + '$\sigma$=%.4f' %params[9] + ' $\mu$=%.3f'%params[4] + ' r=%.3f'%params[5], fontsize=20)
    plt.legend()
    plt.savefig('Plots/inference.png',dpi=500,bbox_inches = "tight")
# %% End
print("--- %f minutes ---" % ((time.time() - start_time)/60.))

""""------------------------------------------------------------
# END
"""
    
