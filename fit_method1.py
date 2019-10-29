''' This file is the first method described in Suess+19a to model the intrinsic M/L 
profile given an observed M/L profile. We assume that the intrinsic M/L gradient
is a power-law function of radius. We generate a bunch of possible M/L slopes, 
model the resulting mass distribution for a galaxy with a given sersic light profile,
convolve both the mass and light profiles with the PSF, then extract a 1D convolved
M/L profile. We use chi^2 minimization to find the model that best matches the data,
and report the M/L gradient and re_mass of that model as the M/L and re_mass of the
galaxy. We also do Monte Carlo simulations to get error bars on the M/L and re_mass.
This file updates the saved galDict created by measureML.py. It also makes a
new savefile that has all of the profiles saved (for plotting later).
'''

# import some functions/libraries that we'll use
from photFuncs import *
from astropy.modeling.models import Sersic2D
from scipy.signal import fftconvolve
from scipy.special import gamma
from scipy.special import gammainc
from collections import namedtuple
from scipy.interpolate import interp1d
import sys
import glob

# import plotting packages and set some pretty plot defaults
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
plt.interactive(True)
sns.set(style="ticks", font_scale=1.5)
sns.set_style({"xtick.direction":u'in', "ytick.direction":u'in', "lines.linewidth":1.5})


############################ FUNCTIONS ############################
'''Define a few functions that we'll use when forward modeling instrinsic M/L gradients. '''

def Ie(Ltot, re, n):
    '''This function returns the light profile Ie for a sersic function, given the
    sersic index "n", half-light radius "re", and total luminosity "Ltot". See
    Graham & Driver 2005, "A consice reference to projected sersic quantities", for
    a full description and lots of math about how sersic functions work.'''
    
    # approximate the value of the bn constant using G&D05 equation (only valid for
    # some n-- if we're not in the right range, don't extrapolate...)
	if not (0.5<= n <= 10):
		return None
	bn = 1.9992*n - 0.3271
        
    # calculate light profile according to G&D05; return profile   
	return(Ltot / (2*np.pi * n * re**2.) * bn**(2.*n)/np.exp(bn) * 1./gamma(2.*n))

def get_rehalf(re, n, ml, edges, lastedge=50):
    '''Given a sersic n and re, as well as a M/L gradient, calculate the
    half-mass radius. We do this using the analytic sersic function instead of
    the 2D mass function above because it avoids issues with pixelation. '''
    
    # generate a 1D sersic profile for the light distribution 
    # over a large range of radii
	rs = np.logspace(-2, 5, 500)
	b = 1.9992*n - 0.3271
	x = b * (rs / re)**(1./n)
	lightProf = gammainc(2*n, x)
    
    # calculate the light *in each annulus* of this distribution
	peran = lightProf - np.insert(lightProf[:-1], 0, 0)
    
    # and multiply by the M/L gradient to turn it into a mass profile
	prof = rs ** (np.log10(ml) / np.log10(lastedge))
    
    # to avoid numeric issues: set mass profile constant below 1 pixel...
	prof[rs < 1] = prof[np.argmin(np.abs(rs-1))] 
    # ... and also set mass profile to be constant after our last measured M/L value
	prof[rs > edges[-1]] = prof[np.argmin(np.abs(rs-edges[-1]))] 
    
    # since this was per-annulus, do a sum to find the total mass profile
	massProf = np.cumsum(prof * peran)

    # return the half-mass radius, as well as the entire mass profile as a 
    # function of radius (for plotting)
	return np.interp(massProf[-1]/2., massProf, rs)

def make_model(n, re, MLs, x0, y0, x, y, dist, edges, lastedge=50):
    ''' This function is where we generate the grid of all possible models for a 
    given galaxy. Given a sersic index "n" and a half-light radius "re",
    we make models of the 2D light distribution of a galaxy with that
    n/re. Then, we apply a range of power-law M/L gradients to turn that light profile 
    into a bunch of theoretical mass profiles. We convolve the 2D mass profiles with 
    the HST F160W PSF to get a set of theoretical 'as-observed' mass profiles, then 
    extract the 1D versions of those mass profiles in annuli. We return the set of both 
    the intrinsic and convolved possible mass profiles for the galaxy.
    "MLs" : list of possible M/L gradients
    "x0, y0" : where to center the galaxy 
    "x, y" : grid of x/y values where we want to evaluate the model
    "edges" : list of the radii of the elliptical apertures we used to calculate the
        M/L gradient. We want to pull out the model M/L at these same radii. In units
        of pixels.
    "dist" : array that has the distance of each pixel in the model from the center
        x0,y0 location. this is used to calculate the actual M/L gradient.
    '''
    
	# make 2D sersic model. re should be given in *pixels*!
	lightModel = Sersic2D(amplitude=1e5, r_eff=re, n=n, x_0=x0, y_0=y0)(x,y)

	# multiply M/L gradients by that light profile to make mass profiles
	massModel = lightModel[:,:, np.newaxis] * \
		dist[:, :, np.newaxis] ** (np.log10(MLs[:-1]) / np.log10(lastedge))
    # get the half-mass radii for all those mass profiles     
	rehalf = np.array([get_rehalf(re, n, ml, edges, lastedge) for ml in MLs])

	# Convolve both the light and mass profiles with the HST PSF
	lightConv = fftconvolve(lightModel, psf, mode='same')
	massConv = np.array([fftconvolve(massModel[:,:,m], psf, mode='same')
		for m in range(len(MLs[:-1]))])
    
    # initialize some arrays that will hold the model light and mass
    # profiles (both in observed & deconvolved space)    
	light = np.zeros((len(edges)))
	mass = np.zeros((len(MLs), len(edges)))
	lightInt = np.zeros((len(edges)))
	massInt = np.zeros((len(MLs), len(edges)))
	edges = np.append(edges, edges[-1]+1)
    
    # measure aperture photometry in annuli for both light and mass
	for an in range(len(edges)-1):
        # make the aperture at this radius
		ap = photutils.CircularAperture((x0,y0), r = edges[an])
        
		# and measure aperture photometry on light profile in that annulus
		light[an] = photutils.aperture_photometry(lightConv, ap)['aperture_sum'][0]
		lightInt[an] = photutils.aperture_photometry(lightModel, ap)['aperture_sum'][0]
        
        # measure aperture photometry for all possible mass maps in that annulus
		for slope_idx in range(len(MLs[:-1])):
			mass[slope_idx, an] = photutils.aperture_photometry(massConv[slope_idx,:,:], \
				ap)['aperture_sum'][0]
			massInt[slope_idx, an] = photutils.aperture_photometry(massModel[:,:,slope_idx], \
				ap)['aperture_sum'][0]
		mass[-1, :] = light
		massInt[-1,:] = lightInt

    # ultimatly we want the profile in *annuli* so just subtract smaller apertures
    # from larger apertures to make the profile in annuli not cumulative distributions
	lightDiff = light - np.insert(light[:-1], 0, 0)
	massDiff = mass - np.insert(mass[:, :-1], 0, np.zeros((len(MLs))), axis=1)
	lightDiffInt = lightInt - np.insert(lightInt[:-1], 0, 0)
	massDiffInt = massInt - np.insert(massInt[:, :-1], 0, np.zeros((len(MLs))), axis=1)
    
    # what we want to return:
    # 1) half-mass radii for each M/L gradient
    # 2) "as-observed" eg convolved M/L profile for each possible M/L gradient
    # 3) intrinsic M/L profile for each possible M/L gradient
    # keep in mind that the "r" axis of these arrays matches our measurements
	return (rehalf, massDiff / lightDiff[np.newaxis, :], massDiffInt / lightDiffInt[np.newaxis, :])

	
##################################### SETUP #####################################
'''Get set up to run this method for modeling the intrinsic M/L from observed
M/L! Want to load in all of the work we've already done as well as the survey
data we'll need to run this file. '''	

# this program should be called command line as "python fit_method1.py group"
# where "group" is the name of the param file & savefile used in measure_ML.py
# if no group is listed, exit
if len(sys.argv) < 2:
	sys.exit('Please list the group as a command-line argument')
group = sys.argv[1]

# open up the savefile from measure_ML.py for this group
# there were a few oddnesses with the way I saved the actual production runs of
# this code for Suess+19a that make the following try/except make sense...
try:
	galDict = np.load('savefiles/'+group+".npz", fix_imports = True, encoding = 'latin1')['galDict'][()]
except FileNotFoundError: # means I had to split dictionary up into separate savefiles
	files = glob.glob('savefiles/'+g+'_'+survey+"*.npz")
	galDict = {}	
	for f in files:
		galDict = {**galDict, **np.load(f)['galDict'][()]}

# make sure we have a name that we'll use to save all of our outputs to later...
savename = 'savefiles/'+group+'_fitresults.npz'

# make sure we grab the psf image for the correct survey
# I think this was updated to read in from the param file, but can't access that code b/c of power outage...
if 'cosmos' in group:
	psf = fits.open('/Users/wren/Surveys/3D-HST/COSMOS/Images/'+
        'cosmos_3dhst_v4.0_wfc3_psf/cosmos_3dhst.v4.0.F160W_psf.fits')[0].data
elif 'uds' in group:
	psf = fits.open('/Users/wren/Surveys/3D-HST/UDS/Images/'+
        'uds_3dhst_v4.0_wfc3_psf/uds_3dhst.v4.0.F160W_psf.fits')[0].data
elif 'cdfs' in group:
	psf = fits.open('/Users/wren/Surveys/3D-HST/GOODS-S/Images/'+
        'goodss_3dhst_v4.0_wfc3_psf/goodss_3dhst.v4.0.F160W_psf.fits')[0].data			

# set up a few model parameters: the maximum radius we want to integrate out to, the edges
# to measure M/L at, and the possible M/L slopes
maxRadius = 50 # pixels
edges = np.linspace(1, maxRadius, maxRadius) # measure once/pixel
MLs = np.array([200, 100, 50, 25, 10, 7.5, 5, 3, 2, 1.5, 1.25, 1.1, .9, .8, .7, .6, .5, .4, .3,
	.2, .1, .05, .025, .01, .005, 1]) # make SURE last one is one (i.e., no slope)
if not MLs[-1] == 1.0:
	sys.exit("last M/L slope is assumed to be constant! Fix this!")

# set up the grid that we'll evaluate the model on
# make sure that the model is centered *between* two pixels to avoid a weird 
# numerical error in the sersic2D function in astropy
rGrid = 150 
x, y = np.meshgrid(np.arange(rGrid), np.arange(rGrid)) # has plenty extra for convolution padding
x0, y0 = x[0][-1]/2., x[0][-1]/2.
# make array w/ distance from each point to center (used to make mass from light grid)
dist = np.zeros(x.shape)
for i in range(x.shape[0]):
	for j in range(x.shape[1]):
		dist[i, j] = np.sqrt((i-x0)**2. + (j-y0)**2.)

# set up a bunch of dictionaries that will hold the model results for each galaxy
MLresults = {}
reResults = {}
chibest = {}
bestModel = {}
bestModel_int = {}
model68 = {}
model95 = {}
model68_int = {}
model95_int = {}


################################## MODEL EACH GALAXY ##################################
'''For each galaxy:
1) make a grid of possible as-observed M/L models given some intrinsic M/L slopes
2) these models are arbitrarily normalized--- scale them up to match the observed M/L
3) find the model that best fits the data (smallest chi^2)
4) Monte Carlo (using measured M/L error) to get error bars 
 '''

for galID in galDict.keys():
    # make a grid of possible M/L profiles
	rmass, model, modelInt = make_model(galDict[galID].n, galDict[galID].re/pixScale, MLs,
		x0, y0, x, y, dist, galDict[galID].edges)
    
    # scale the mass models up to the same scale as the M/L observations    
	scale = np.sum(model * np.broadcast_to(galDict[galID].bestML_f160, \
		model.shape), axis=-1, keepdims=True) /		\
		np.sum(model **2., axis=-1, keepdims=True)
	model_scaled = model * scale  
	modelInt_scaled = modelInt * scale

	# calculate chi^2 b/t all models and the data
    # this needs an error, but our error bars on M/L are asymmetric.
    # treat this simplistically: if model < data, use the lower error bars;
    # if model > data, use the upper error bars
    
    # create "errs" that has upper/lower errors as appropriate
	upper = model_scaled > galDict[galID].bestML_f160
	lower = np.invert(upper)
	errs = np.zeros((upper.shape))
	errs[upper] = (np.broadcast_to(galDict[galID].MLerr_f160[1,:], upper.shape))[upper]
	errs[lower] = (np.broadcast_to(galDict[galID].MLerr_f160[0,:], upper.shape))[lower]
    
    # and calculate the chi^2
	chiSq = np.sum((((galDict[galID].bestML_f160 - model_scaled)**2.) / errs**2.), axis=-1)
    
    # the best M/L and best half-mass radius are the one that minimizes chi^2
	bestML = MLs[np.argmin(chiSq)]
	bestRe = rmass[np.argmin(chiSq)]
    
    # also keep track of the best chi^2 to evaluate later goodness of fit
	chibest[galID] = np.min(chiSq) / (len(galDict[galID].bestML_f160) - 1)

	# now we want to do some Monte Carlo simulations to determine error bars on the
    # best-fit M/L and half-mass radius.
    # in these simulations, we want to let n_light and re_light vary within their
    # error bars, and also let the observed M/L gradient vary w/i its error bars
    
    # set up: make arrays to hold simulation results
	nsim = 200
	simML = []
	simRe = []
    
    # for each simulation, generate random draw from the data & fit it
	for sim in range(nsim):
		# get random light_re, light_n, and bestML
		modN = np.random.normal(galDict[galID].n, galDict[galID].dn)
		modRe = np.random.normal(galDict[galID].re/pixScale, galDict[galID].dre/pixScale) # in pixels
		modML = np.array([doubGauss(galDict[galID].bestML_f160[i],
			galDict[galID].MLerr_f160[1][i], galDict[galID].MLerr_f160[0][i]) for
			i in range(len(galDict[galID].bestML_f160))])

		# make new model grid (as above)
		modRmass, modModel, modModelInt = make_model(modN, modRe, MLs, x0, y0, x, y, dist, galDict[galID].edges)
		modscale = np.sum(modModel * np.broadcast_to(modML, modModel.shape), \
			axis=-1, keepdims=True) / np.sum(model **2., axis=-1, keepdims=True)
		modModel_scaled = modModel * modscale

		# re-calculate chi sq for this monte carlo draw
		modupper = modModel_scaled > modML
		modlower = np.invert(modupper)
		moderrs = np.zeros((modupper.shape))
		moderrs[modupper] = (np.broadcast_to(galDict[galID].MLerr_f160[1,:], modupper.shape))[modupper]
		moderrs[modlower] = (np.broadcast_to(galDict[galID].MLerr_f160[0,:], modupper.shape))[modlower]
		modchiSq = np.sum((((modML - modModel_scaled)**2.) / errs**2.), axis=-1)
		
        # record the best-fit M/L and r_mass for this simulation
        simML.append(MLs[np.argmin(modchiSq)])
		simRe.append(modRmass[np.argmin(modchiSq)])

    # record the 16/84 and 2.5/97.5 (e.g., 1 & 2 sigma) errors for the monte carlo simulations
	MLresults[galID] = np.array([bestML, np.percentile(simML, 16),
		np.percentile(simML, 84), np.percentile(simML, 2.5),
		np.percentile(simML, 97.5)])
	reResults[galID] = np.array([bestRe, np.percentile(simRe, 16),
		np.percentile(simRe, 84), np.percentile(simRe, 2.5),
		np.percentile(simRe, 97.5)])
        
    # record the best overall model in case we want to plot it later
	bestModel[galID] = model_scaled[np.argmin(chiSq)]
	bestModel_int[galID] = modelInt_scaled[np.argmin(chiSq)]

	# record the 68/95% convolved models
	mod_68 = model_scaled[np.where((MLs >= np.percentile(simML, 16)) & \
		(MLs <= np.percentile(simML, 84)))[0], :]
	mod_95 = model_scaled[np.where((MLs >= np.percentile(simML, 2.5)) & \
		(MLs <= np.percentile(simML, 97.5)))[0], :]
	model68[galID] = np.array((np.amin(mod_68, axis=0), np.amax(mod_68, axis=0)))
	model95[galID] = np.array((np.amin(mod_95, axis=0), np.amax(mod_95, axis=0)))

	# record the 68/95th intrinsic models
	mod_68 = modelInt_scaled[np.where((MLs >= np.percentile(simML, 16)) & \
		(MLs <= np.percentile(simML, 84)))[0], :]
	mod_95 = modelInt_scaled[np.where((MLs >= np.percentile(simML, 2.5)) & \
		(MLs <= np.percentile(simML, 97.5)))[0], :]
	model68_int[galID] = np.array((np.amin(mod_68, axis=0), np.amax(mod_68, axis=0)))
	model95_int[galID] = np.array((np.amin(mod_95, axis=0), np.amax(mod_95, axis=0)))

	# put M/L and re results into sensible units 
    # M/L is in terms of M/L(2re_light) / M/L(re_light)
	MLresults[galID] = 2**(np.log10(MLresults[galID]) / np.log10(50)) 
    # re is in r_mass / r_light (instead of pixels)
	reResults[galID] = reResults[galID] / (galDict[galID].re/0.06) #re_mass / re_light
    
	# put re_mass results into galDict 
    # we want to include re *in kpc!* and the chi^2 of the best fit
	galDict[galID].re_fitres = np.append(reResults[galID] * galDict[galID].re / \
        cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value,
		 chibest[galID]) 

    # update the user
	print('Made model for galaxy '+str(galID))
	
	
# re-save the galDict since now it has the fit results for re_mass
np.savez('savefiles/'+group+'.npz', galDict=galDict) 

# also save all the big dictionaries-- we'll want them for plotting
np.savez(savename, MLresults=MLresults, reResults=reResults, chibest=chibest, \
	bestModel=bestModel, bestModel_int=bestModel_int, model68=model68, model95=model95, \
	model68_int=model68_int, model95_int=model95_int)



