# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import numpy as np
from scipy import special

def bin2d(array, diameter, mode='median'):
    arshape = array.shape
    array_rs = array.reshape([arshape[0]*arshape[1], -1])
    ind_range = np.expand_dims(np.arange(array_rs.shape[0]),1)
    ind_roi = np.repeat(np.arange(-arshape[1]*(diameter//2),arshape[1]*(diameter//2)+1,arshape[1]), diameter)
    ind_roi += np.tile(np.arange(-(diameter//2), diameter//2+1), diameter)
    pad = arshape[1]*(diameter//2) + diameter//2
    ind_complete = ind_range + ind_roi + pad
    pad = arshape[1]*(diameter//2) + diameter//2
    array_rs = np.pad(array_rs, ((pad,pad),(0,0)), mode='constant', constant_values=float('nan'))
    if mode == 'mean':
        array_binned = np.nanmean(array_rs[ind_complete,:],1)
    elif mode == 'median':
        array_binned = np.nanmedian(array_rs[ind_complete,:],1)
    array_binned = np.squeeze(array_binned.reshape([arshape[0], arshape[1], -1]))
    return array_binned


def adaptive_median(array, diameter, threshold):
    array_out = array.copy()
    array_median = bin2d(array, diameter, mode='median')
    sel = (array < (array_median*threshold[0])) | (array > (array_median*threshold[1]))
    array_out[sel] = array_median[sel]
    array_out[np.isnan(array)] = array_median[np.isnan(array)]
    return array_out

def pileup_corr(data, pileup_factor):
    data_corr = -np.log(1-np.clip(data*pileup_factor, None, 255-1e-12)/255)*255/pileup_factor
    return data_corr

def irf_simple(t, t0, gate_fwhm, sigma):
    '''
    IRF approximation: Gaussian laser pulse integrated with sharp edges
    '''
    out = np.sqrt(np.pi/2) * sigma * special.erf((t-t0+gate_fwhm)/np.sqrt(2)/sigma)
    out -= np.sqrt(np.pi/2) * sigma * special.erf((t-t0)/np.sqrt(2)/sigma)
    return out

def decay1_simple2(t, t0, gate_fwhm, sigma, A, tau):
    '''
    Less simple decay approximation: Errorfunction decay integrated with sharp edges
    '''
    out = (special.erf(sigma/np.sqrt(2)/tau - (t-t0+gate_fwhm)/np.sqrt(2)/sigma) - 1)*np.exp(-(t-t0+gate_fwhm)/tau)
    out += np.exp(-sigma**2/2/tau**2)*special.erf((t-t0+gate_fwhm)/np.sqrt(2)/sigma)
    out -= (special.erf(sigma/np.sqrt(2)/tau - (t-t0)/np.sqrt(2)/sigma) - 1)*np.exp(-(t-t0)/tau)
    out -= np.exp(-sigma**2/2/tau**2)*special.erf((t-t0)/np.sqrt(2)/sigma)
    out = out*A*tau/2*np.exp(sigma**2/2/tau**2)    
    return out

def decay_mult_simple2(t, t0, gate_fwhm, sigma, A, r, tau):
    '''
    Less simple decay approximation: Errorfunction decay integrated with sharp edges
    '''
    out = 0
    rat = np.hstack([r, 1-np.sum(r)])
    for I in range(len(tau)):
        out += decay1_simple2(t, t0, gate_fwhm, sigma, rat[I], tau[I])
    out = A*out
    return out

def decay1_min(t0, A, tau, fwhm, sigma, t, decay):
    model = decay1_simple2(t, t0, fwhm, sigma, A, tau) + 10
    loss = (decay+10) * np.log((decay+10) / model) - (decay + 10 - model)
    return np.sum(loss)

def decay2_min(t0, gate_fwhm, sigma, A, r, tau1, tau2, t, decay):
    model = decay_mult_simple2(t, t0, gate_fwhm, sigma, A, r, [tau1, tau2]) + 10
    loss = (decay+10) * np.log((decay+10) / model) - (decay + 10 - model)
    return np.sum(loss)