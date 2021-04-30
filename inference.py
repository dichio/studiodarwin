#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 2020
Last revised: Wed Mar 17 2021
@author: Vito
"""

""""------------------------------------------------------------
Inferring interaction from chi_ij by nMF
input: chi_ij (correlation between i and j) -- mean chi2
output: fitness interaction fij_nMF
"""
import numpy as np
import sys, math
from scipy import linalg

def nMF_Fij(chi2):
    n,l = chi2.shape
    c_inv = linalg.inv(chi2)
    J_nMF = -c_inv
    for i0 in range(n):
        J_nMF[i0,i0] =0
    return J_nMF

def infer_formulas(inveps, J, n):
    L,L = J.shape
    J = J - np.diag(np.diag(J))
    if n=='I' :
        fI = np.zeros((L,L))
        for i in range(L):
            for j in range(L):
                if i != j :
                    fI[i,j]=inveps[i,j]*J[i,j]
        return fI                
    elif n=='II' :
        fI = np.zeros((L,L))
        for i in range(L):
            for j in range(L):
                if i != j :
                    fI[i,j]=inveps[i,j]*J[i,j]
        fII = np.zeros((L,L))
        corr = np.zeros((L,L))                
        for i in range(L):
            for j in range(L):
                if i != j :            
                    for k in range(L):
                        corr[i,j] += fI[i,k]*J[j,k]+fI[j,k]*J[i,k]
                        fII[i,j]=fI[i,j]-corr[i,j] 
        return fII
    else:
        sys.exit("Check orders I or II in infer_formulas !")

        
def c_ij_eval(params):
    cij = np.zeros((params[0], params[0]))
    for i in range(params[0]):
        for j in range(params[0]):
            cij[i, j] = 0.5 * (1.0 - math.exp(-2.0 * params[6] * abs(i - j)))
    rc = params[5]*cij
    inveps = 4.0*params[4] + rc
    return [rc,inveps]

def err_epsilon(f_rec, f_true):
    """ 
    This is eq. (24) in Zeng,Aurell: 'Inferring genetic fitness from genomic data' (2020) 
    """
    n, n = np.shape(f_true)
    diff = f_rec - f_true
    rmse = np.sqrt(np.sum(diff**2))/np.sqrt(np.sum(f_true**2))
    return rmse


