# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

import numpy as np
from glob import glob
import scipy.optimize as opt
import time
import os
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, disk


from fit_functions import adaptive_median, pileup_corr, decay1_min, decay1_simple2

import matplotlib.pyplot as plt

param_irf = np.load('./reflection_irf_fitresult.npy')
bg = np.load('./background_gated_avg_cps.npy')

param_irf = np.transpose(adaptive_median(param_irf.T, 3, [0.5,1.5]))

plot = True
pileup_factor = 3.5
bounds_mle = ([3,7],
              [0,np.inf],
              [0.1,20])


folder = './gated_20ms_40cts_0'
mask_circ = tifffile.imread(folder + '_mask.tif')
parameters = eval(open(folder + '/parameters.txt').read())
t = np.arange(parameters['num_steps'])*parameters['gate_step']*1e-3
files = sorted(glob(folder + '/data/file*.npy'))

for I, file in enumerate(files):
    fname = os.path.split(file)[1][:-4]
    print(fname)
    outfile = folder + 'fitresult_{:s}_corr{:.0f}.npy'.format(fname, pileup_factor*10)
    
    if not os.path.isfile(outfile):

        data = np.load(file).astype(float)
        
        # background correction
        data_corr = (data[0,...] - bg*parameters['exposure']/1e3)
        
        # pileup correction
        data_corr = pileup_corr(data_corr, pileup_factor)
        intensity = np.sum(data_corr, 0)
        
        if plot:
            plt.figure()
            plt.imshow(intensity)
            plt.colorbar()
            plt.show()
        
        # segmentation mask
        intensity_blur = ndimage.gaussian_filter(intensity, 10)
        threshold = threshold_otsu(intensity_blur)
        mask = intensity_blur > threshold
        mask = binary_dilation(mask, disk(10))
        mask = mask * mask_circ.astype(bool)

        if plot:        
            plt.figure()
            plt.imshow(intensity*mask)
            plt.colorbar()
            plt.show()
        
        fitresult = np.zeros([5,512,512])
        fitresult[0,:,:] = intensity
        fitresult[1,:,:] = data_corr[np.argmax(np.sum(data_corr, (1,2))),:,:]

        if plot:
            plt.figure()
            plt.plot(np.sum(data_corr, (1,2)))
            plt.show()
        
        num_fits = 0
        num_nans = 0
        
        tic = time.time()
        for i in range(512):
            for k in range(512):
                decay = data_corr[:,i,k]
                if mask[i,k] and np.sum(decay) > 100:
                    t0 = np.mean(param_irf[1, i//4, k//4])
                    fwhm = np.mean(param_irf[2, i//4, k//4])
                    sigma = np.mean(param_irf[3, i//4, k//4])
                    pin = [t0, np.max(decay)/3, 3]
                    minfunc = lambda x: decay1_min(x[0], x[1], x[2], fwhm, sigma, t, decay)
                    result = opt.minimize(minfunc, pin, bounds=bounds_mle, method='L-BFGS-B', options={'maxfun':200})
                    par = result['x']
                    if result['success']:
                        fitresult[2:,i,k] = par
                        num_fits += 1
                    else:
                        fitresult[2:,i,k] = float('nan')
                        num_nans += 1

        
        np.save(outfile, fitresult)
        
        print('{:d} successfull fits and {:d} unsuccessfull in {:.1f} s'.format(num_fits, num_nans, time.time()-tic))
        
        if plot:
            plt.figure()
            plt.imshow(fitresult[4,:,:])
            plt.colorbar()
            plt.show()
            
            plt.figure()
            plt.hist(np.ravel(fitresult[-1,:,:]), bins=np.linspace(1,4,100))
            plt.show()