# Calculating Galaxy Half-Mass Radii

Code for calculating the half-mass radii of galaxies used in [Suess+19a](https://arxiv.org/pdf/1904.10992.pdf). As described in detail in the paper, we use three different methodologies to calculate half-mass radii from multi-band, PSF-matched imaging. The first two are separate methods for modeling intrinsic M/L from an observed (ie, convolved with the PSF) M/L profile. The third is an implementation of the Szomoru+11,12,13 method.

## Measuring the M/L profile 
`measureML.py` is used to measure the M/L profile from multi-band, PSF-matched imaging. This piece of code:
* reads in an input ".param" file listing filepaths, galaxy IDs, redshifts, etc
* measures aperture photometry in elliptical annuli for each galaxy
* runs the stellar population synthesis (SPS) fitting code FAST to model the spectral energy distribution of each annulus
* adjusts the best-fit SPS parameters for each annulus such that the sum of the models for each annulus in a galaxy matches the integrated light of that galaxy in all available filters. This allows us to incorporate information from bands where we don't have resolved imaging, and is especially important for correctly modeling the near-IR flux from the galaxy.
* reports a best-fit mass and M/L in each annulus
* does Monte Carlo simulations to get error bars on the mass and M/L in each annulus
* saves output as a numpy savefile 

Many of the functions used in this script are stored in `photFuncs.py`, both for readability and so they can be referenced by other files.

This code requires the following packages: numpy, math, scipy, os, itertools, sys, subprocess, glob, matplotlib, seaborn, time, astropy, photutils. 
This code also requires that you have downloaded the 3D-HST data set (both catalogs and imaging).
Finally, it requires that you have a working FAST installation (and thus an IDL license) as well as a working EAZY installation.

The code is run command-line with the following syntax: `python measureML.py path_to_param_file/paramfile.param`. I typically run it in `ipython` using the magic `%run` command.


## Method 1 half-mass radii: forward model method
`fit_method1.py` is used to implement the first method for measuring half-mass radii described in Suess+19a; this is our preferred method for measuring half-mass radii. This piece of code:
* reads in the same input ".param" file as above. 
* reads in the as-observed M/L profiles measured using `measureML.py` above
* assumes that the intrinsic M/L profile is a power-law function of radius, and generates a series of potential M/L profiles for each galaxy
* makes a model for the intrinsic 2D light distribution of the galaxy (from its best-fit sersic parameters)
* multiplies each possible M/L gradient by this light distribution to produce possible mass maps for the galaxy
* convolves both the light map and all possible mass maps with the PSF
* extracts 1D profiles of light and mass in annuli, divides these profiles to get a series of possible M/L profiles
* compares all modeled convolved-space M/L profiles with the observed M/L profile to find the best M/L slope
* uses the intrinsic mass profile of the best-fit model to find the half-mass radius of the galaxy
* does Monte Carlo simulations to get error bars on the half-mass radius

The code is run command-line with the following syntax: `python measureML.py paramfile` (and assumes that the paramfile is located in a directory called `input_files`...). I typically run it in `ipython` using the magic `%run` command.


## Method 2 half-mass radii: GALFIT method
`fit_method2.py` is run to calculate half-mass radii using the second method described in Suess+19a. This piece of code:
* reads in the same input ".param" file as above. 
* reads in the as-observed M/L profiles measured using `measureML.py` above
* makes a cutout of the F160W image around each galaxy
* multiplies this F160W cutout by a (smoothed) version of the measured M/L profile to make a mass map
* runs `GALFIT` to fit this mass map with a Sersic profile
* reads in the best-fit `GALFIT` results, including the half-mass radius
* does Monte Carlo simulations to get error bars on the half-mass radius

In addition to the dependencies listed for `measureML.py`, this code requires that you have a working `GALFIT` installation.

This code is run command-line with the same syntax as `fit_method1.py`.

## Method 3 half-mass radii: Szomoru method
`fit_method3.py` replicates the Szomoru+11,12,13 method for calculating half-mass radii. This piece of code:
* reads in the same input ".param" file as above. 
* creates a cutout around each galaxy in each filter with resolved imaging
* runs `GALFIT` on each cutout to get the best-fit sersic profile in that band; construct the corresponding 1D sersic profile
* measure the flux in elliptical annuli on the `GALFIT` residual image; add this residual back onto the 1D sersic profile for 0-10kpc (no residual correction past 10kpc because the residual is typically quite noisy/mostly sky here)
* use `EAZY` to interpolate between the 1D light profiles in each resolved imaging band to get 1D profiles in the rest-frame u and g SDSS bands
* subtract the u and g profiles to get u-g color profile
* use the full 3D-HST dataset to create a linear fit to get M/L_g given u-g color
* use that fit to turn the u-g profile to a M/L_g profile
* multiply by the L_g profile to get the galaxy's mass profile
* find the half-mass radius
* perturb u-g color profile according to its error bars; do Monte Carlo simulations to get error bars on the half-mass radius

This code is run command-line with the same syntax as `fit_method1.py`.


## Citation & Contact
If you use any pieces of the above code, please cite [Suess+19a](https://arxiv.org/pdf/1904.10992.pdf). Please feel free to email suess (at) berkeley (dot) edu with any questions.
