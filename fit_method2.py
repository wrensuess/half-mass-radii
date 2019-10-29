'''This file implements the second method of modeling the intrinsic M/L gradient
from an observed M/L gradient described in Suess+19. We make a cutout of the F160W
image around the galaxy, then multiply it by a (smoothed version of) the measured
observed-space M/L gradient. This creates a mass map. We fit this mass map with
GALFIT to get a best-fit re_mass and n_mass. We calculate error bars on these
quantities by Monte Carlo simulations. We perturb the observed M/L gradient
according to its error bars, make a new mass map, and re-fit it with galfit.
his file updates the saved galDict created by measureML.py. It also makes a
new savefile that has additional saved properties (in case we need them later).
'''

# import some libraries and functions we'll need
from photFuncs import *
from writeGalfit import *
from astropy.modeling.models import Sersic2D
from scipy.signal import fftconvolve
from scipy.special import gamma
from scipy.special import gammainc
from collections import namedtuple
from scipy.interpolate import interp1d
import sys
import glob
from scipy import interpolate

# some plotting imports & settings for pretty plots
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
plt.interactive(True)
sns.set(style="ticks", font_scale=1.5)
sns.set_style({"xtick.direction":u'in', "ytick.direction":u'in', "lines.linewidth":1.5})

##################################### SETUP #####################################
'''Get set up to run this method for modeling the intrinsic M/L from observed
M/L! Want to load in all of the work we've already done as well as the survey
data we'll need to run this file. '''

# this program should be called command line as "python fit_method2.py group"
# where "group" is the name of the param file & savefile used in measure_ML.py
# if no group is listed, exit
if len(sys.argv) < 2:
	sys.exit('Please list the group as a command-line argument')
group = sys.argv[1]

# use function in photFuncs.py to parse in the input file and check to make sure
# that all the paths actually exist.
survey, IDs, zs, filterResPath, translatePath, galfitPath, catalogPath, images, \
     filters, rmFASTgrid, pixScale, psf, fastPath, fastDir, fastOut, library, \
     nsim, savedict, imPaths = readInputFile(inputfile)

#### Make translate file  ####
# keys are the names of the filters (as they appear in the 3D-HST survey); 
# values are the central wavelength of the filter and the EAZY/FAST filter number
translate = makeTranslateDict(filterResPath, translatePath)

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

# make sure that we have a name to save our results to once we're done
savename = 'savefiles/'+group+'_fitresults_galfit.npz'

# not all IDs from the input file actually had successful observed M/L calculations
# make sure that we're only including the ones that actually worked (avoids key errors...)
IDs = np.array(list(galDict.keys())) 

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

# open up the segmentation, data, and weight images so that we can make cutouts around each galaxy
hduSeg = fits.open(imPaths['seg'])
hduData = fits.open(imPaths[images[0]])
wcsData = WCS(hduData[0].header)
paData = 0.0 # from F160W
hduWht = fits.open(imPaths[images[0]+'_wht'])

# open up the 3D-HST catalog (will use to get a few integrated photometry points)
cat = fits.open(catalogPath)

# use header info to get data -> physical unit conversion
photflam = hduData[0].header['photflam']

# in some bands in UDS, the photflam value is zero-- for
# these values, use the COSMOS values instead
if photflam == 0:
	print('Using photflam from COSMOS for filter '+filterName)
	if images[0] == 'F140W': photflam = 1.4737148e-20
	elif images[0] == 'F160W': photflam = 1.9275602e-20
	else: sys.exit('No photflam found.')

# we'll also need to feed galfit the photometric zeropoint of the survey
# (for sky calculations). These values are pulled from the Skelton+14
# survey paper.	
if survey == 'cosmos':
	if images[0] == 'F160W':
		zpt = 25.956
	elif images[0] == 'F140W':
		zpt = 26.465
elif survey == 'uds':
	if images[0] == 'F160W':
		zpt = 26.452
	elif images[0] == 'F140W':
		zpt = 26.452
elif survey == 'cdfs':
	if images[0] == 'F160W':
		zpt = 25.946
	elif images[0] == 'F140W':
		zpt =  26.452							  
else:
	print('Need to grab right zeropoints for this survey!')
	raise SystemExit	
	
# make a (very basic) header for galfit runs
# (b/c we're supplying sigma image, only need an arbitrary exptime)
exptime = 1000.
header = fits.Header()
header['EXPTIME'] = exptime


################################## MODEL EACH GALAXY ##################################
'''For each galaxy:
1) Make a cutout of the detection-band image around each galaxy. 
2) Multiply that cutout by a (smoothed) version of the measured M/L gradient to
    create an "as-observed" mass map
3) Use galfit to fit this mass map and get a half-mass radius and (mass) sersic index.
    Galfit is a stand-alone program that is run from the command line; do the same trick
    we played before with FAST and use python to write out the input file, run the
    code, then read in the outputs.
4) Do Monte Carlo simulations to get error bars on the half-mass radius.
 '''

# make a dictionary to hold all of our results
galfit_results = {}	 

# for each galaxy:
for galID in IDs:	
    
	####################### Make a mass map #######################
    
	# make cutouts of the data, segmentation map, and weight map
	pixY, pixX = galDict[galID].get_pix(wcsData)
	dat = cutout(hduData[0].data, pixX, pixY, pscale = galDict[galID].pscale)
	whtDat = cutout(hduWht[0].data, pixX, pixY, pscale = galDict[galID].pscale)
	segDat = cutout(hduSeg[0].data, pixX, pixY, pscale = galDict[galID].pscale)
    
    # make a 'mask' array that masks out *only* other galaxies (not sky)
	mask = (segDat!=galDict[galID].id) & (segDat!=0) 	  

	# make array w/ distance from each point to center (used to make mass from light grid)
	distEllipse = np.zeros(dat.shape)
	paRad = (-galDict[galID].pa)* np.pi / 180.
	for i in range(dat.shape[0]):
		for j in range(dat.shape[1]):
			distEllipse[i, j] = np.sqrt(((i-dist.shape[0]/2.)*np.cos(paRad) + \
                (j-dist.shape[0]/2.)*np.sin(paRad))**2. + \
				1./galDict[galID].qConv**2. * ((i-dist.shape[0]/2.)*np.sin(paRad) - \
                (j-dist.shape[0]/2.)*np.cos(paRad))**2.)

    # the M/L profile that we measured is essentially a step function-- if we multiply this
    # by the light profile, it makes a map with discontinuities that galfit freaks out
    # over. So first, we want to smooth the observed M/L profile to get rid of these edge effects
    # if the galaxy is super small, pad it to make sure galfit has enough pixels to work with
	if galDict[galID].edges[-1]+galDict[galID].edges[0] < 100:
		xfit = np.append(galDict[galID].edges, np.linspace(galDict[galID].edges[-1]+galDict[galID].edges[0], 100))
		yfit = np.append(galDict[galID].bestML_f160, np.zeros(50)+galDict[galID].bestML_f160[-1]) 
	# if the galaxy is small, don't bother padding
    else:
		xfit = galDict[galID].edges
		yfit = galDict[galID].bestML_f160	
    # use a cubic spline interpolation    
	spfit = interpolate.splrep(xfit, yfit, s=0)
    
    # make the mass map: measured light cutout multiplied by the smoothed M/L map
	mass = dat * interpolate.splev(distEllipse, spfit, der=0)
	    
	# replace any NaNs in the mass map & weight map so galfit doesn't get mad
	mass[np.isnan(mass)] = 0
	whtDat[np.isnan(mass)] = 0
	
	# write out fits files for the mass map, sigma (1/sqrt(weight)) map, and mask. 
    # both the data and weight maps need to be in units of *counts* fits for data. 
    # we do this by multiplying both of them by an exposure time
	fits.writeto('galfitInputs/mass_'+str(galID)+'_dat.fits',
		data=mass*exptime, header = header, overwrite=True)
	fits.writeto('galfitInputs/mass_'+str(galID)+'_wht.fits',
		data=interpolate.splev(distEllipse, spfit, der=0) * exptime *
		np.sqrt(np.abs(dat/exptime) + np.divide(1, whtDat, 
		out=np.zeros_like(whtDat), where=np.sqrt(whtDat)!=0)),
		header = header, overwrite=True)
	fits.writeto('galfitInputs/mass_'+str(galID)+'_mask.fits',
		data=mask.astype(int), header = header, overwrite=True)

    # make sure we get the total catalog magnitude (to write into a galfit constraint file)
	mag = 25 - 5/2. * np.log10(cat[1].data[galDict[galID].id - 1]['f_F160W'])
    
    # and the expected position (at the center of our cutout)
	pos = [galDict[galID].pscale+pixY%1, galDict[galID].pscale+pixX%1]
   
	####################### Write GALFIT param file #######################
	galfit_write_param('galfitInputs/mass_'+str(galID)+'.input', "mass_"+str(galID)+'_dat.fits', \
		"mass_"+str(galID)+'_output.fits', "mass_"+str(galID)+'_wht.fits', "psf_"+survey+".fits", \
		"mass_"+str(galID)+'_mask.fits', "mass_"+str(galID)+'.constraints', mass.shape, zpt, \
		str(galDict[galID].pscale+pixX%1)[:7], str(galDict[galID].pscale+pixY%1)[:7], str(mag)[:7], \
		str(galDict[galID].re/pixScale)[:7], str(galDict[galID].n)[:7], str(galDict[galID].q)[:7], \
        str(galDict[galID].pa)[:7])
        
	# also write the constraint file (constraints described in detail in Suess+19a)
	galfit_write_constraints("galfitInputs/mass_"+str(galID)+'.constraints', mag, nmin='0.2', \
		nmax='8.0', remin='0.3', remax='400')

	####################### Run GALFIT #######################
	os.chdir("galfitInputs")
	return_code = subprocess.call('/Users/wren/galfit/galfit mass_'+ \
	    str(galID)+'.input > mass_'+str(galID)+'.log', shell=True)
	os.chdir("..")

	####################### Read in GALFIT results #######################
    # make sure the fit actually ran...
	if not os.path.isfile('galfitInputs/mass_'+str(galID)+'_output.fits'):
		print('No output file found for galaxy '+str(galID))
		continue
        
    # and then read in the output .fits file    
	hduOut = fits.open('galfitInputs/mass_'+str(galID)+'_output.fits')

	# check output flags-- if it crashed, move on
	if ('2' in hduOut[2].header['flags'].split(' ')) or \
		('1' in hduOut[2].header['flags'].split(' ')):
		print('Bad galfit flag for galaxy '+str(galID))
		continue

    # record the best-fit re and n
	try:
		galfit_results[galID] = np.zeros((5)) # best, 16%, 84%, 2.5%, 97.5%
		galfit_results[galID][0] = (float(hduOut[2].header['1_re'].split()[0]))
		n = float(hduOut[2].header['1_n'].split()[0])
	except ValueError:
		continue
			
	####################### Monte Carlo for error bars #######################
	# initialize an array to hold the results of the MC simulations
    res = []
    
    # for each simulation, repeat the above process
	for i in range(200):
        # first perturb our measured M/L profile according to its error bars
		yfit = doubGauss(galDict[galID].bestML_f160, galDict[galID].MLerr_f160[1,:], \
            galDict[galID].MLerr_f160[0,:])
            
        # as above, smooth the M/L using a cubic spline  
		if galDict[galID].edges[-1]+galDict[galID].edges[0] < 100:
			yfit = np.append(yfit, np.zeros(50)+yfit[-1]) # keep M/L constant after last measurement
		spfit = interpolate.splrep(xfit, yfit, s=0)
		mass = dat * interpolate.splev(distEllipse, spfit, der=0)

		# replace NaNs in files so galfit doesn't get mad
		mass[np.isnan(mass)] = 0
		whtDat[np.isnan(mass)] = 0

		# write fits for data-- weight map doesn't need to be updated
        # since it's the same as before
		fits.writeto('galfitInputs/mass_'+str(galID)+'_dat.fits',
			data=mass*exptime, header = header, overwrite=True)
			
		# run galfit
		os.chdir("galfitInputs")
		return_code = subprocess.call('/Users/wren/galfit/galfit mass_'+ \
		    str(galID)+'.input > mass_'+str(galID)+'.log', shell=True)
		os.chdir("..")

		# read in galfit results
		if not os.path.isfile('galfitInputs/mass_'+str(galID)+'_output.fits'):
			continue
		hduOut = fits.open('galfitInputs/mass_'+str(galID)+'_output.fits')

		# check output flags-- if it crashed, move on
		if ('2' in hduOut[2].header['flags'].split(' ')) or \
			('1' in hduOut[2].header['flags'].split(' ')):
			continue
		
        # and store the MC draw of the r_mass for this simulation
        try:
			res.append(float(hduOut[2].header['1_re'].split()[0]))
		except ValueError:
			continue
	
    # now, calculate the 1sigma & 2sigma error bars on re for this galaxy		
	res = np.array(res)	
    galfit_results[galID] = np.percentile(res, [16, 84, 2.5, 97.5])
    
	# update the user:
    print('Calculated re_mass for galaxy '+string(galID))
		
	# add galfit to galDict so we don't have to re-load
    # make sure it's in units of kpc!!
	galDict[galID].re_gf = galfit_results[galID] * pixScale / \
        cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value	
		
	# remove galfit log files (tiny, but there are so many of them!)
	for f in os.listdir('galfitInputs/'):
		if f.startswith('galfit'):
			os.remove('galfitInputs/'+f)		

# save the dictionary with our detailed galfit results
np.savez('savefiles/'+group+'_fitgalfit.npz', galfit_results=galfit_results)

# since we updated the galDict with our half-mass radii, re-save it
np.savez('savefiles/'+group+'.npz', galDict=galDict) 

# close the files that we opened above	
hduData.close()
hduSeg.close()
hduWht.close()
