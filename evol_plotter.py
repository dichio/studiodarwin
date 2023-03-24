#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 15:35:03 2022

@author: vito.dichio
"""

import os,sys
import numpy as np
os.chdir('/Users/vito.dichio/ownCloud/mypastprojects/2020_studiodarwin')
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
import pandas as pd
import math
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
if __name__ == "__main__": 
    print(sys.argv)
    
def cm(x): 
    # From fucking inches to cm
    return x*0.3937
    

#%% Fig. 2

 # upper tringular matrix vectorized
def triv(x):
    return x[np.triu_indices(len(x),k=1)]

def myscinot(v):
    myticks = np.array([])
    for x in v:
        if x !=0:
            scale = math.floor(math.log(abs(x), 10))
            myticks = np.append(myticks,'{:,.1f}'.format(x*10**(-scale)) + '$_{{10}^{'+str(int(scale))+'}}$')
        else:
            myticks = np.append(myticks,'0.0')
    return myticks

def RMSE(x,y):
    return (np.sum((x-y)**2)/np.sum(x**2))**(1./2)

label_size = 8
char_size = 8
mydpi = 600
cols = ["#495C83","#201E20","#E0A96D"]

ftrue = triv(np.loadtxt('./output/fig2/f_testing.txt'))
fMF = triv(np.loadtxt('./output/fig2/f_nMF.txt'))
fPLM = triv(np.loadtxt('./output/fig2/f_plm.txt')) 

df = pd.DataFrame(data = np.column_stack((ftrue,fMF,fPLM)), columns = ["ftrue","fMF", "fPLM"])

sns.set_style("darkgrid")

fig, ax = plt.subplots(figsize=(cm(12), cm(8)))

ax.plot([0, 1], [0, 1], transform=ax.transAxes, lw=1., color = cols[0], ls='--')

sp1 = sns.scatterplot(data=df,x="ftrue",y="fMF",color=cols[1],alpha=0.9,marker='h',label='MF; $\ \epsilon=%.2f$' %RMSE(ftrue,fMF))
sp2 = sns.scatterplot(data=df,x="ftrue",y="fPLM",color=cols[-1],alpha=0.9,marker='o',label='PLM;$\ \epsilon=%.2f$' %RMSE(ftrue,fPLM))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.ylim([-1e-2,1e-2])
plt.xlim([-1e-2,1e-2])
sp1.set_yticklabels(myscinot(sp1.get_yticks()), size = char_size)
sp1.set_xticklabels(myscinot(sp1.get_xticks()), size = char_size)

ax.legend(loc=2,fontsize=char_size,ncol=1)

sp1.set(xlabel=r'$f_{ij}^{\ true}$', ylabel=r'$f_{ij}^{\ *}$')

plt.savefig('Plots/fig2.pdf',dpi=mydpi,bbox_inches = "tight")
plt.show()


#%% Fig. 3

data = scipy.io.loadmat('./output/fig3/Data_MuR_PLM.mat')
murPLM = np.rot90(data['Data_MuR_PLM'])
data = scipy.io.loadmat('./output/fig3/Data_SigmaR_PLM.mat')
srPLM = np.rot90(data['Data_SigmaR_PLM'])

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 

label_size = 8
char_size = 6
mydpi = 600

myvmin = min(np.min(murPLM),np.min(srPLM))
myvmax = max(np.max(murPLM),np.max(srPLM))
mycmap = sns.cubehelix_palette(start=.2,rot=-.3,dark=0.05, light=.95,n_colors=50,reverse=True)

rlabels = ['{:,.0f}'.format(x) + '$_{{10}^{-1}}$' for x in np.linspace(10.,0.0,6)]; rlabels[-1] = str('0'); rlabels[0] = str('1$_{{10}^{+0}}$')
rlabels = ['{:,.1f}'.format(1.0),'{:,.1f}'.format(0.8),'{:,.1f}'.format(0.6),'{:,.1f}'.format(0.4),'{:,.1f}'.format(0.2),'{:,.1f}'.format(0.0)] 
mlabels = ['{:,.0f}'.format(x) + '$_{{10}^{-2}}$' for x in np.linspace(1.,10,4)]; mlabels[-1] = str('1$_{{10}^{-1}}$')
slabels = ['{:,.0f}'.format(x) + '$_{{10}^{-3}}$' for x in np.linspace(1.,10,4)]; slabels[-1] = str('1$_{{10}^{-2}}$')

fig, ax = plt.subplots(1, 2, figsize=(4,2.5),sharex=False, sharey=True)
cbar_ax = fig.add_axes([.95,.2,.02,.6])



g1 = sns.heatmap(murPLM,
            ax=ax[0],
            cbar=True,
            vmin=myvmin,vmax=myvmax,
            cbar_ax = cbar_ax,
            cmap=mycmap,
            yticklabels=rlabels,
            xticklabels=mlabels,
            )

g2 = sns.heatmap(srPLM,
            ax=ax[1],
            cbar=True,
            vmin=myvmin,vmax=myvmax,
            cbar_ax = cbar_ax,
            cmap=mycmap,
            yticklabels=rlabels,
            xticklabels=slabels)   

g1.set_yticks(np.linspace(0,20,6)+0.5)
g1.set_yticklabels(g1.get_yticklabels(), rotation = 0, fontsize = char_size)

g1.set_xticks(np.linspace(2,20,4)-0.5)
g1.set_xticklabels(g1.get_xticklabels(),rotation = 0, fontsize = char_size)

g2.set_xticks(np.linspace(2,20,4)-0.5)
g2.set_xticklabels(g2.get_xticklabels(),rotation = 0, fontsize = char_size)

g1.set(xlabel=r'$\mu$', ylabel='r')
g2.set(xlabel=r'$\sigma_e$',ylabel="")

g1.figure.axes[0].xaxis.label.set_size(label_size)
g1.figure.axes[0].yaxis.label.set_size(label_size)
g2.figure.axes[1].xaxis.label.set_size(label_size)
cbar_ax.tick_params(labelsize=6)

plt.savefig('Plots/fig3.pdf',dpi=mydpi,bbox_inches = "tight")



#%% Fig. 4

data = scipy.io.loadmat('./output/fig4/Data_MuR_KNS.mat')
murKNS = data['Data_MuR_KNS']

data = scipy.io.loadmat('./output/fig4/Data_MuR_GA.mat')
murGA = data['Data_MuR_GA']

data = scipy.io.loadmat('./output/fig4/Data_SigmaR_KNS.mat')
srKNS =  data['Data_SigmaR_KNS']

data = scipy.io.loadmat('./output/fig4/Data_SigmaR_GA.mat')
srGA = data['Data_SigmaR_GA']

def myscinot(v):
    myticks = np.array([])
    for x in v:
        if x !=0:
            scale = math.floor(math.log(abs(x), 10))
            myticks = np.append(myticks,'{:,.1f}'.format(x*10**(-scale)) + '$_{{10}^{'+str(int(scale))+'}}$')
        else:
            myticks = np.append(myticks,'0.0')
    return myticks


# plot-params
sns.set_style("dark")
size_scale = 120
n_colors = 100
#palette = sns.diverging_palette(250, 30, l=50, n = n_colors, sep=1, s=99, center='light')
palette = sns.color_palette("magma",n_colors)
label_size = 8
char_size = 8
mydpi = 600
mrk = '8'

# Create meshgrid
ny, nx = np.shape(murGA)
xcoor = np.arange(nx); ycoor = np.arange(ny)
xx, yy = np.meshgrid(xcoor,ycoor) 

# Create colors - errorKNS-errorGA
demur = murKNS - murGA 
demur_vec = np.reshape(demur,(nx*ny)).tolist()

desr = srKNS - srGA 
desr_vec = np.reshape(desr,(nx*ny)).tolist()

cl_min, cl_max = [min(np.min(desr),np.min(demur)), max(np.max(desr),np.max(demur))]


def value_to_color(val):  
    val_position = float((val - cl_min)) / (cl_max - cl_min) 
    ind = int(val_position * (n_colors - 1))
    return palette[ind]

def value_to_color_sym(val):
    bound = 0.5
    if val>bound:
        return palette[-1]
    elif val<-bound:
        return palette[0]
    else:
        val_position = float((val + bound)) / (2*bound) 
        ind = int(val_position * (n_colors - 1))
        return palette[ind]

# Start plotting
fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(cm(14),cm(14)),sharex=True, sharey=False)

sc1 = ax[0].scatter(
        x=xx, 
        y=yy, 
        s=(1-murGA) * size_scale, # Vector of square sizes, proportional to size parameter
        c=list(map(value_to_color, demur_vec)),
        marker=mrk
    )

sc2 = ax[1].scatter(
        x=xx, 
        y=yy, 
        s=(1-srGA) * size_scale, # Vector of square sizes, proportional to size parameter
        c=list(map(value_to_color, desr_vec)),
        marker=mrk
    )

#  ticks
rlabels = [str('{:,.1f}'.format(0)),str('$0.05$'),str('{:,.1f}'.format(0.1)),str('$0.15$'),str('{:,.1f}'.format(0.2)),str('$0.25$'),str('{:,.1f}'.format(0.3)), str('$0.35$'),str('{:,.1f}'.format(0.4)),str('$0.45$'),str('{:,.1f}'.format(0.5)),str('$0.55$'),str('{:,.1f}'.format(0.6)),str('$0.65$'), str('{:,.1f}'.format(0.7)),str('$0.75$'),str('{:,.1f}'.format(0.8)),str('$0.85$'),str('{:,.1f}'.format(0.9)),str('$0.95$'),str('{:,.1f}'.format(1.0))]
mlabels = ['{:,.1f}'.format(x) + '$_{{10}^{-2}}$' for x in np.linspace(.5,5.0,10)]; 
slabels = ['{:,.1f}'.format(x) + '$_{{10}^{-3}}$' for x in np.linspace(.5,5.0,10)];     

x_to_num = {p[1]:p[0] for p in enumerate(rlabels)}        
ax[1].set_xticks([x_to_num[v] for v in rlabels])
ax[1].set_xticklabels(rlabels, rotation=0, horizontalalignment='center',fontsize = char_size)

xticks = ax[1].xaxis.get_major_ticks()
for i in range(len(rlabels)):
    if i%2!=0:
        xticks[i].label1.set_visible(False)

y_to_num = {p[1]:(p[0]) for p in enumerate(mlabels)}
ax[0].set_yticks([y_to_num[v] for v in mlabels])
ax[0].set_yticklabels(mlabels, rotation=0, horizontalalignment='right',fontsize = char_size)

y_to_num = {p[1]:(p[0]) for p in enumerate(slabels)}
ax[1].set_yticks([y_to_num[v] for v in slabels])
ax[1].set_yticklabels(slabels, rotation=0, horizontalalignment='right',fontsize = char_size)

yticks1 = ax[0].yaxis.get_major_ticks()
yticks2 = ax[1].yaxis.get_major_ticks()
for i in range(len(mlabels)):
    if i%2==0:
        yticks1[i].label1.set_visible(False)
        yticks2[i].label1.set_visible(False)
        
ax[0].grid(False, 'major')
ax[0].grid(True, 'minor')
ax[0].set_xticks([t + 0.5 for t in ax[0].get_xticks()], minor=True)
ax[0].set_yticks([t + 0.5 for t in ax[0].get_yticks()], minor=True)

##
ax[1].grid(False, 'major')
ax[1].grid(True, 'minor')
ax[1].set_xticks([t - 0.5 for t in ax[0].get_xticks()], minor=True)
ax[1].set_yticks([t + 0.5 for t in ax[0].get_yticks()], minor=True)
#
#
ax[0].set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
ax[0].set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
ax[1].set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

# Axes labels
ax[0].set(xlabel='', ylabel=r'$\mu$')
ax[1].set(xlabel= r'$r$', ylabel=r'$\sigma_e$')

ax[0].figure.axes[0].xaxis.label.set_size(label_size)
ax[1].figure.axes[0].yaxis.label.set_size(label_size)
ax[1].figure.axes[1].xaxis.label.set_size(label_size) 

# Size Legend
gll = plt.scatter([],[], s=0.1*size_scale , marker=mrk, color='#5f549e')
gl = plt.scatter([],[], s=0.5*size_scale , marker=mrk, color='#5f549e')
ga = plt.scatter([],[], s=1.0*size_scale , marker=mrk, color='#5f549e')

leg = plt.legend((gll,gl,ga),
       ('0.1', '0.5', '1.0'),
       scatterpoints=1,
       title = r'$\alpha^{\ GA}$',
#       title_fontsize=label_size,
       loc='lower left',
       ncol=1,
       fontsize=label_size,
       bbox_to_anchor=(1.10,0.3),
       labelspacing=2,
       frameon=False,)
plt.setp(leg.get_title(),fontsize=label_size)

# Color bar
cbar_ax = fig.add_axes([1.,.55,0.02,.25])

my_cmap = ListedColormap(sns.color_palette("magma",n_colors).as_hex())
sm = plt.cm.ScalarMappable(norm=Normalize(cl_min,cl_max),cmap='magma')
sm.set_array([])

clb = ax[0].figure.colorbar(sm,
                      cax = cbar_ax,
                      orientation = 'vertical',
                      )
clb.ax.set_title(r'$\alpha^{\ GA}-\alpha^{\ KNS}$', fontsize = label_size)
clb.ax.set_yticklabels(clb.ax.get_yticklabels(), fontsize=label_size)

# Problem w/ colorbar prevents from exporting in pdf!
plt.savefig('Plots/fig4.png',dpi=mydpi,bbox_inches = "tight")

#%% Fig. 8
sns.set_style("white")
label_size = 8
char_size = 6
ps = 25
mydpi = 600

cols = ["#0008C1","#1CD6CE","#4E944F","#D61C4E"]

NRC = np.load('output/fig8/NRC.npy')
NRC = np.delete(NRC,[15,27,39,52,56],0)
QLE = np.load('output/fig8/QLE.npy')
QLE = np.delete(QLE,[5,14,22,29,37,6,15,23,30,38,7,16,31,45,50],0)
inst = np.load('output/fig8/inst.npy')
inst = np.delete(inst,[11,17,8,10,23],0)

fig, ax = plt.subplots(figsize=(cm(15),cm(5)))
ax.scatter(QLE[:,1], QLE[:,0], s=ps,marker='o',c=cols[0],alpha=.75,lw=.5,edgecolor=cols[0],label='QLE')
ax.scatter(NRC[:,1], NRC[:,0], s=ps,marker='o',c=cols[2],alpha=.75,lw=.5,edgecolor=cols[2],label='NRC')
ax.scatter(inst[:,1], inst[:,0], s=ps,marker='o',c=cols[3],alpha=.75,lw=.5,edgecolor=cols[3],label='QLE$\leftrightarrow$NRC')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(bbox_to_anchor=(0.8,1.25),fontsize=label_size,ncol=3)

#plt.title('Stability' + '\n' + 'T=10000',fontsize=char_size)
plt.ylabel('$N$',fontsize=label_size, rotation = 0)
ax.yaxis.set_label_coords(.0, +1.075)
ax.xaxis.set_label_coords(1.025, +.1)
plt.xlabel('$\sigma_e$',fontsize=label_size)
ax.ticklabel_format(style='scientific',axis='x',scilimits=(0,0),labelsize = char_size)
ax.xaxis.get_offset_text().set_fontsize(char_size)

plt.tick_params(labelsize=label_size)
ax.set_yscale('log', basey=2)
#plt.grid(lw=.1)
plt.xlim([0.008,0.052])
plt.ylim([80,3000])
plt.savefig('Plots/phsp.pdf',dpi=mydpi,bbox_inches = "tight")
plt.show()







