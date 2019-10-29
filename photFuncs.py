'''
Functions used to calculate & fit radial M/L gradients in galaxies.
Stored here for easy importing.
'''

# make sure we've got all the packages we need
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import photutils
import math
import subprocess
import scipy.io
import itertools
from astropy.cosmology import FlatLambdaCDM
import os

# set up a standard cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# a few constants we'll need at some point.
mpc = 3.08568e24	# 1 Mpc in cm
Lsun = 3.9e33 # erg/s-- total solar luminosity
lamg = 4.7025e+03 # central wavelength of SDSS g band filter

def Flam(fnu, lam):
	'''Flux converter: given a wavelength, translates from F_nu to F_lam. 
    Lambda should be given in *angstroms*. Uses equation nu*f_nu = lambda * f_lambda.'''
	return 1./(3.33e4 * lam**2.) * fnu
    
def Fnu_arb_zpt(flam, lam, zpt):
	'''Flux converter: given a wavelength *and an arbitrary flux zeropoint*,
    translates from F_lam to F_nu. Again, uses the equation nu*f_nu = lambda * f_lambda.'''
	return (lam**2/2.998e18*flam)*10**(2/5.*(zpt+48.57))    

def Fnu25(flam, lam):
	'''Special case of the above: go from F_lam to F_nu with zeropoint +25. This
    is useful b/c all catalog fluxes are given with this zeropoint.'''
    return Fnu_arb_zpt(flam, lam, 25)

def flux2lum(f, spec, z):
	'''Convert flux to luminosity. "f" is a SPECIFIC flux (i.e., either f_lambda or f_nu).
	"spec" is either lambda, if f is f_lambda, or nu if f is f_nu. basically, f is multiplied
	by spec to get a flux in erg/s/cm^2 to convert to a luminosity. The value returned is in
	units of Lsun.'''
	return (np.array(f)*spec * 4*np.pi * (cosmo.luminosity_distance(z).value * mpc)**2. / Lsun)
    
def readInputFile(inputfile):
    '''This function reads in and parses the input file ('inputfile' given as a string to the
    location of the file, e.g. "inputdir/inputs.param"). It checks to make sure that it
    can read all of the variables we'll need later, like the survey name, IDs and redshifts,
    paths to the survey catalogs and images, etc. Right now it is set up to be a bit too
    specific to the formatting of the input files I was generating algorithmically for
    Suess+19a,b. Future update will instead read in a .json file that can be a little more
    flexible in terms of exact formatting of the input file... '''
    
    # read lines of the input file
    with open(inputfile) as f:
    	lines = f.readlines()
    lines = [i.rstrip().split(' = ') for i in lines if (i[0] != '#') & (i[:2] != '\n')] # strip out comments, etc

    # and parse them out...
    # the following code will check for each variable that we need to be in the
    # input file. It's really sensitive to formatting of the input file, and
    # will be updated soon to be more general...
    try: survey = [i for i in lines if i[0] == 'survey'][0][1]
    except: sys.exit('Cannot read survey')

    # 3D-HST catalog ID numbers
    try: IDs = [int(j) for j in ([i for i in lines if i[0] == 'IDs'][0][1]).split(', ')]
    except: sys.exit('Cannot read IDs')

    # redshifts (put in manually b/c I used ZFOURGE redshifts not 3D-HST redshifts)
    try: zs = [float(j) for j in ([i for i in lines if i[0] == 'zs'][0][1]).split(', ')]
    except: sys.exit('Cannot read zs')

    # filter res file for FAST
    try: filterResPath = [i for i in lines if i[0] == 'filterResPath'][0][1]
    except: sys.exit('Cannot read filter res path')
    if not os.path.isfile(filterResPath):
    	sys.exit('No file for filter res at '+filterResPath)

    # translate directory that goes between filter numbers and names
    # again, used for FAST
    try: translatePath = [i for i in lines if i[0] == 'translatePath'][0][1]
    except: sys.exit('Cannot read translate path')
    if not os.path.isfile(translatePath):
    	sys.exit('No file for translate at '+translatePath)

    # path to van der Wel+ galfit catalogs for the survey
    try: galfitPath = [i for i in lines if i[0] == 'galfitPath'][0][1]
    except: sys.exit('Cannot read galfit path')
    if not os.path.isfile(galfitPath):
    	sys.exit('No file for galfit at '+galfitPath)

    # path to the 3D-HST catalog
    try: catalogPath = [i for i in lines if i[0] == 'catalogPath'][0][1]
    except: sys.exit('Cannot read catalog path')
    if not os.path.isfile(catalogPath):
    	sys.exit('No file for catalog at '+catalogPath)

    # path to all PSF-convolved images. Detection band is assumed
    # to be the first band that's listed.
    try:
    	images = [i for i in lines if i[0] == 'images'][0][1].split(' #')[0].split(', ')
    	detectionBand = images[0]
    except: sys.exit('Cannot read images')

    # path to the filter curve definitions for the survey
    try:
    	filters = [i for i in lines if i[0] == 'filters'][0][1].split(' #')[0].split(', ')
    except: sys.exit('Cannot read filters')

    # boolean option to carry around the FAST grids (should basically always be False)
    try:
        rmFASTgrid = bool([i for i in lines if i[0] == 'rmFASTgrid'][0][1])
    except: sys.exit('Cannot read if I should remove FAST grid after running')    

    # pixel scale of the 3D-HST images (from Skelton+)
    try: pixScale = float([i for i in lines if i[0] == 'pixScale'][0][1])
    except: sys.exit('Cannot read pixel scale')

    # PSF scale (again, from Skelton+)
    try: psf = float([i for i in lines if i[0] == 'psf'][0][1].split(' #')[0])
    except: sys.exit('Cannot read psf scale')

    # FAST results for the whole survey (provided by 3D-HST)
    try: fastPath = [i for i in lines if i[0] == 'fastPath'][0][1]
    except: sys.exit('Cannot read FAST path')
    if not os.path.isfile(fastPath):
    	sys.exit('No file for FAST at '+fastPath)

    # make sure there's a directory for FAST to run in
    try: fastDir = [i for i in lines if i[0] == 'fastDir'][0][1]
    except: sys.exit('Cannot read FAST directory')
    if not os.path.isdir(fastDir):
    	sys.exit('No FAST directory at '+fastDir)

    # FAST output location
    try: fastOut = [i for i in lines if i[0] == 'fastOut'][0][1]
    except: sys.exit('Cannot read FAST output directory')

    # where to find the FAST libraries
    try: library = [i for i in lines if i[0] == 'library'][0][1][:4]
    except: sys.exit('Cannot read library')

    # how many times to run Monte Carlo simulations for M/L error bars
    try: nsim = int([i for i in lines if i[0] == 'nsim'][0][1])
    except: sys.exit('Cannot read nsim')

    # what we should call the savefil at the end
    try: savedict = [i for i in lines if i[0] == 'savedict'][0][1]
    except: sys.exit('Cannot read dictionary save name')

    # make a dictionary that has the path to each image in the survey.
    imPaths = {}
    # first check the SExtractor segmentation image exists
    try: imPaths['seg'] = [i for i in lines if i[0] == 'path_seg'][0][1]
    except: sys.exit('Cannot read segmentation path')
    # then check that both the data and weight (e.g., inverse square error)
    # image exist for each filter we said was in this survey. 
    for fil in images:
    	try:
    		imPaths[fil] = [i for i in lines if i[0] == 'path_'+fil][0][1]
    		imPaths[fil+'_wht'] = [i for i in lines if i[0] == 'path_'+fil+'_wht'][0][1]
    	except:
    		sys.exit('Cannot read image or wht path for '+fil)
    	if not (os.path.isfile(imPaths[fil]) and os.path.isfile(imPaths[fil+'_wht'])):
    		sys.exit('No file found for filter '+fil)
            
def makeTranslateDict(filterResPath, translatePath):
    '''Make a dictionary that translates between filter
    numbers (used by FAST and EAZY) and central wavelengths.
    Uses the standard EAZY filter res file and also the
    translate file provided by the survey.'''
    
    # generate list of all filter central wavelengths
    fil = np.genfromtxt(filterResPath, usecols=1)
    row = np.where(np.isnan(fil)==True)[0]
    filList = []
    with open(filterResPath) as fd:
    	for n, line in enumerate(fd):
    		if n in row:
    			filList.append(line.split())

    # open translate file 
    transFile = np.genfromtxt(translatePath, dtype='str')
    
    # make a dictionary that contains the translate info we need for each filter
    # keys are the names of the filters (as they appear in the 3D-HST survey); 
    # values are the central wavelength of the filter and the EAZY/FAST filter number
    translate = {}
    for i in range(len(transFile))[::2]:
        row = filList[int(transFile[i][1][1:])-1]
        translate[transFile[i][0][2:].upper()] = [float(row[row.index('lambda_c=')+1]), \
            transFile[i][1]]          
        

def cutout(datArr, pixX, pixY, pscale):
	'''Given a (large) array, make a smaller cutout around the object of interest. The 
    cutout is centered at (pixX, pixY) and is pscale x pscale pixels large.'''
	return datArr[int(pixX)-pscale : int(pixX)+pscale+1, int(pixY)-pscale : int(pixY)+pscale+1]

def emptyApError(fil, area, wht, pscale, survey):
	'''Function for calculating the error in an empty aperture (in data units). 
    Equation is from Skelton+14. This needs to be multiplied by the 'photflam' 
    keyword in the fits header to transform from data units to physical units.'''
    
    # different filters have different extinction factors. These are taken from Skelton+14.
	if fil == 'F606W':
		A = 0.019
	elif fil == 'F814W':
		A = 0.029
	elif fil == 'F814WCAND':
		A = 0.029	 
	elif fil == 'F125W':
		A = 0.014
	elif fil == 'F140W':
		A = 0.012
	elif fil == 'F160W':
		A = 0.010
	elif fil == 'F435W':
		A =	 0.028
	elif fil == 'F775W':
		A =	 0.013
	elif fil == 'F850LP':
		A = 0.010			 
	else:
		print('Extinction factor not found. Assuming no extinction.')
		A = 0
        
    # find the average value of the weight array at this position    
	weight = np.mean(wht[pscale-2:pscale+3, pscale-2:pscale+3])
    
    # make sure the weight map actually has a value
	if weight > 0:
        # coefficients depend on survey; see Skelton+14 for details.
		if survey == 'cosmos':
			return 0.525 * area**0.64 * (1/np.sqrt(weight)) * 10**(0.4*A)
		elif survey == 'uds':
			return 0.45 * area**0.655 * (1/np.sqrt(weight)) * 10**(0.4*A)
		elif survey == 'cdfs':
			return 0.45 * area**0.655 * (1/np.sqrt(weight)) * 10**(0.4*A)
	
    # if no weight map value (shouldn't happen, but good to check),
    # then return a nonsense empty aperture error.
    return -99.

def write_FAST_catalog(pathOut, filters, translate, images, galDict, IDs):
	'''Writes a FAST input catalog for all galaxies. Because FAST doesn't accept IDs
	that are longer than 7 characters, instead use ID that's the number in the ID
	list. ie, if galaxy 10298 is the first in IDs, it's ID in the FAST catalog
	will be 1, followed by the annulus number (so 10, 11, 12, 13, ... 19, 110,
	111, etc). Make sure the numbers are non-overlapping...
    The format for the FAST catalog follows the standard fast.param examples
    that come with the FAST software.'''

	# open the output file and write a header
	outfile = open(pathOut, 'w')
	outfile.write('# ID\tz_spec\t')
	for fil in filters:
		outfile.write(translate[fil][1] + '\tE' + translate[fil][1][1:] + '\t')
	outfile.write('\n')
    
	# write the row for each galaxy
    # each 'gal' object holds all its own photometry, so let the class
    # function do the hard part of writing out all of the values.
	for num, galID in enumerate(IDs):
		if num < 9: # want to avoid confusion between 110 (1-10) and 110 (11-0)
			lines = galDict[galID].write_FAST(filters, translate, images, num+1)
		else:
			lines = galDict[galID].write_FAST(filters, translate, images, (num+1)*10)
		for l in lines:
			outfile.write(l + '\n')
	outfile.close()

def closest(val, arr):
	'''Return index of array that's closest to a given input value'''
	return np.argmin(abs(arr - val))

def doubGauss(center, upper, lower):
	'''Select random value from a "double gaussian"-- one center, but diff
	sigma on top and bottom. Upper is upper_error; lower is lower_error.
    This is useful for monte carlo trials for variables that have 
    asymmetric error bars (e.g., most of our values...).
    Mathematically, we can do this just by re-scaling a normally distributed
    random variable.'''
	ran = np.random.normal(loc=0., scale=1.)
    # if the random number is on the upper half of the distribution, scale 
    # by that error bar.
	if ran > 0:
		return(center + ran*(upper))
    # if not, scale by the lower error bar.    
	return(center + ran*(lower))


############################
class gal(object):
    '''Class for each galaxy. This both carries around all of the interesting info
    about each galaxy (basic properties, catalog values, measured aperture photometry,
    derived properties) and has some class functions that we'll use later on. '''

	def __init__(self, survey, idnum, z, galfit, catalog, fast, pixScale, psf, translate):
		'''
        When we initialize the galaxy, want to make sure we have all of the basic 
        info that will be important later on.
		Survey should be the survey the galaxy comes from (COSMOS, CDFS, or UDS),
		and idnum should be the ID number that galaxy has in that survey catalog
        (3D-HST). Also want the redshift of the galaxy to tote around.
		'''

		# basic stuff
		self.survey = survey
		self.id = idnum
		self.z = z 

		# read in galfit properties from van der Wel+14 catalog
		self._get_galfit(galfit, pixScale, psf)
        # van der Wel catalog has flag=3 if galfit didn't run for this galaxy
		if self.galfitflag == 3:
			print('No galfit results for galaxy '+str(self.id))

		# get catalog fluxes in integrated bands from 3D-HST catalog file.
		self._get_catalogFlux(catalog, translate)

        # now, calculate the edges and the annuli for this galaxy (in *pixels*).
        # described in detail in get_edges function below.
		self._get_edges(pixScale, psf, maxRadius=50)

		# initialize things that will be filled in later
        # these are all values that we'll compute at some point during the
        # following analysis.
		self.photometry = {}
		self.corr = 1.
		self.grid = []
		self.chiGrid = []
		self.chiThresh = []
		self.massGrid = []
		self.bestPos = []  # Av, logAge, logTau
		self.lowerPos = [] # Av, logAge, logTau
		self.upperPos = [] # Av, logAge, logTau
		self.bestMasses = np.zeros((self.nAnnuli))
		self.massErrors = np.zeros((2, self.nAnnuli))
		self.Lg = []
		self.LgErr = []
		self.bestML = []
		self.MLerr = []

	def _get_galfit(self, galfit, pixScale, psf):
		'''
        Given a galfit catalog ('galfit'), adds galfit properties to class instance. 
        Used in initialization.
		'''
        # locate the row
		row = np.where(galfit[:,0] == self.id)[0][0]
        # and extract all the properties of the galfit fit
		self.ra, self.dec, self.galfitflag, self.mag, self.dmag, self.re, self.dre, \
			self.n, self.dn, self.q, self.dq, self.pa, self.dpa = galfit[row,1:-1]

		# make "convolved" ax ratio
        # basically, we want to smear the b/a out by the pixel scale.
		self.a = np.sqrt((self.re/pixScale)**2. + (psf/pixScale)**2.)
		self.b = np.sqrt((self.re/pixScale*self.q)**2. + (psf/pixScale)**2.)
		self.qConv = self.b/self.a

	def _get_catalogFlux(self, cat, translate):
		''' Adds 3D-HST catalog fluxes and errors to class instance. Used in
		initialization.
		'''
        # fluxes and errors will be stored in a dictionary.
        # keys are the name of the filter; values are the fluxes in that filter.
        # the filter names should match the 'translate' dictionary so we can get
        # pivot wavelengths for converting f_nu <-> f_lam
		self.catalogFlux = {}
		self.catalogError = {}
        
        # find the right row
		idx = np.where(cat[1].data['id'] == self.id)[0][0]
        
        # and get all of the filters that we have data for (different for each survey)
		filterNames = [i for i in cat[1].data.dtype.names if i.startswith('f_')]
        
        # store fluxes and errors for each filter.
		for fname in filterNames:
			filCenter = translate[fname[2:].upper()][0]
			self.catalogFlux[fname] = Flam(cat[1].data[fname][idx]*10.**(-6.44), filCenter)
			self.catalogError[fname] = Flam(cat[1].data['e'+fname[1:]][idx]*10.**(-6.44), filCenter)
		
        # check use flag
		self.flag = cat[1].data['use_phot'][idx]

	def _get_edges(self, pixScale, psf, maxRadius=50):
		'''Sets number of annuli and edges that will be used for calculating aperture
        photometry. We want each annulus to be spaced 1 PSF FWHM apart.
        When this is first called, it makes more annuli that will eventually be used-- 
        when the aperture photometry is calculated in the detection band, it will trim 
        any annuli that have too low S/N'''
        # start with large number of annuli
		self.nAnnuli = int(np.ceil(np.sqrt(self.re**2. + psf**2.)*maxRadius / psf))
		# edges are spaced evenly from 0 outwards by one PSF FWHM
        self.edges = np.linspace(0,
			np.sqrt(self.re**2. + psf**2.)*maxRadius/pixScale, self.nAnnuli+1)[1:]
		# 'pscale' sets the cutout size. Make sure that cutouts are at least
        # 50 pixels on a side; can be larger if galaxies are larger.
        self.pscale =  max(50, int(self.edges[-1]+20))

	def get_pix(self, wFil):
		'''Given a WCS file, returns the (pixY, pixX) corresponding
        to a specific ra/dec. Make sure that we use the right convention
        for this!'''
		pixY, pixX = wFil.wcs_world2pix(self.ra, self.dec, 1)
		return pixY-1, pixX-1

	def make_apertureList(self, wcs, paIm):
		''' Make a list of photutils apertures to calculate aperture photometry.
        Need to give a WCS header object so that we can get the x,y location
        of the galaxy in the data. Then, make an elliptical aperture for each
        edge listed for this galaxy. The apertures should have a b/a equal to
        the convolved axis ratio we calculated above. Make sure to check the 
        theta to make sure it matches the convention we expect. '''
		# find (x,y) location of galaxy in the arry
        pixY, pixX = self.get_pix(wcs)
        
        # initialize aperture list
		apertureList = []
		
        # for each radius, make an elliptical aperture.
        # make sure to keep the sub-pixel shifts (galaxy centers not exactly 
        # on a single pixel)
        for r in self.edges:
			apertureList.append(photutils.EllipticalAperture(positions=
			(self.pscale+pixY%1, self.pscale+pixX%1),
			a=r, b=r*self.qConv,
			theta=math.radians(paIm)-math.radians(self.pa)))
            
        # return the whole list to use later    
		return apertureList

	def calcPhotometry_detectionband(self, filterName, photflam, data, wcs, weight, seg, paIm, SNthresh=10.):
		'''Calculates aperture photometry for the detection band. The detection band
        is a different function from the rest of the bands because we'll use it to 
        determine the total number of annuli. Essentially, we want to use the full
        list of apertures we calculated above, but *stop calculating* after we reach
        a given S/N threshold. We use a threshold of 10 in Suess+19, but this
        can be tuned depending on the use. Each annulus is 1 PSF FWHM wide.'''

        # make sure the catalog photometry is good for this object
		if self.flag != 1:
			print('Could not calculate photometry for '+str(self.id)+', flag = '+str(self.flag))
			return -99

		# make cutouts of the of the data, weight map, and segmentation map around this galaxy
		pixY, pixX = self.get_pix(wcs)
		dat = cutout(data, pixX, pixY, pscale = self.pscale)
		whtDat = cutout(weight, pixX, pixY, pscale = self.pscale)
		segDat = cutout(seg, pixX, pixY, pscale = self.pscale)
        
        # explicitly mask out anything that SExtractor identified as belonging to another galaxy.
        # note that this doesn't mask out anything SExtractor identifies as sky (good, 
        # b/c it's usually too conservative in its flux threshold for our purposes....)
		mask = (segDat!=self.id) & (segDat!=0) 

		# make a long list of potential apertures; will trim later
		apList = self.make_apertureList(wcs, paIm)

		# initialize lists to store aperture photometry and its error
		self.photometry[filterName] = []
		self.photometry[filterName+'_err'] = []

		# treat the first annulus separately: makes our stopping criterion below easier
		ann=0
		area = apList[ann].area() # built in photutils area-calculating function
        # do aperture photometry on the mask to calculate the total non-masked area...
		maskArea = photutils.aperture_photometry(mask, apList[ann])['aperture_sum'][0]
		# ... and use it to make a correction factor to scale up the flux
        corr = area / (area - maskArea)
		flux = photutils.aperture_photometry(dat, apList[ann], mask=mask)['aperture_sum'][0] \
			* corr * photflam # calculate flux in physical units
		error = emptyApError(filterName, area, whtDat, self.pscale, self.survey) * photflam
        # add flux and error to the list of measured aperture photometry
		self.photometry[filterName].append(flux)
		self.photometry[filterName+'_err'].append(error)

		# following annuli: quit when S/N reaches the threshhold
		while (self.photometry[filterName][-1] / self.photometry[filterName+'_err'][-1]) > SNthresh:
			ann = ann+1 # increment counter
            # area of an annulus is the area of this ellipse minus the ellipse inside it
			area = apList[ann].area() - apList[ann-1].area() 
            # same for the mask
			maskArea = (photutils.aperture_photometry(mask, apList[ann])['aperture_sum'][0] -
				photutils.aperture_photometry(mask, apList[ann-1])['aperture_sum'][0])
			corr = area / (area - maskArea)
            # and for the flux
			flux = (photutils.aperture_photometry(dat, apList[ann], mask=mask)['aperture_sum'][0] -
				photutils.aperture_photometry(dat, apList[ann-1], mask=mask)['aperture_sum'][0]) \
				* corr * photflam
            # empty ap area only depends on the total area of the elliptical aperture    
			error = emptyApError(filterName, area, whtDat, self.pscale, self.survey) * photflam
			# make sure that our measured flux is finite, and append it to the list
            if np.isfinite(flux):
				self.photometry[filterName].append(flux)
				self.photometry[filterName+'_err'].append(error)
			else:
				self.photometry[filterName].append(np.nan)
				self.photometry[filterName+'_err'].append(np.nan)

		# trim extra annuli / edges that had S/N below the threshold
		# but first, make sure that we actually have annuli to work with...
		if ann==0:
			return(-99)
		self.nAnnuli = ann # update total number of annuli with the max we got to
		self.edges = self.edges[:ann] # trim unnecessary edges
		self.pscale = max(50, int(self.edges[-1]+20)) # and update the cutout scale if necessary

		# trim the last photometric point (the one that failed S/N check)
		self.photometry[filterName] = self.photometry[filterName][:-1]
		self.photometry[filterName+'_err'] = self.photometry[filterName+'_err'][:-1]

	def calcPhotometry(self, filterName, photflam, data, wcs, weight, seg, paIm):
		'''Calculates aperture photometry for all annuli in one image.
		Data, weight, and seg should be the full hdu[x].data arrays '''
        
        # make sure the catalog photometry is good for this object
		if self.flag != 1:
			print('Could not calculate photometry for '+str(self.id)+', flag = '+str(self.flag))
			return np.nan

		# make cutouts of the of the data, weight map, and segmentation map around this galaxy
		pixY, pixX = self.get_pix(wcs)
		dat = cutout(data, pixX, pixY, pscale = self.pscale)
		whtDat = cutout(weight, pixX, pixY, pscale = self.pscale)
		segDat = cutout(seg, pixX, pixY, pscale = self.pscale)
		mask = (segDat!=self.id) & (segDat!=0) # mask out other galaxies

		# make list of apertures
		apList = self.make_apertureList(wcs, paIm)

		# don't measure photometry if the wht array is zero (ie, no exposures here)
		if not np.sum(whtDat):
			self.photometry[filterName] = [np.nan for i in range(self.nAnnuli)]
			self.photometry[filterName+'_err'] = [np.nan for i in range(self.nAnnuli)]
			return 0

		# initialize lists to store aperture photometry and its error
		self.photometry[filterName] = []
		self.photometry[filterName+'_err'] = []
        
        # and go actually measure aperture photometry in each annulus
		for ann in range(self.nAnnuli):
            # first annulus is just an ellipse (not an annulus)
			if ann == 0:
				area = apList[ann].area()
				maskArea = photutils.aperture_photometry(mask, apList[ann])['aperture_sum'][0]
				corr = area / (area - maskArea)
				flux = photutils.aperture_photometry(dat, apList[ann], mask=mask)['aperture_sum'][0] \
					* corr * photflam
			# otherwise, subtract off flux from the ellipse before this one to make an elliptical annulus
            else:
				area = apList[ann].area() - apList[ann-1].area()
				maskArea = (photutils.aperture_photometry(mask, apList[ann])['aperture_sum'][0] -
					photutils.aperture_photometry(mask, apList[ann-1])['aperture_sum'][0])
				corr = area / (area - maskArea)
				flux = (photutils.aperture_photometry(dat, apList[ann], mask=mask)['aperture_sum'][0] -
					photutils.aperture_photometry(dat, apList[ann-1], mask=mask)['aperture_sum'][0]) \
					* corr * photflam
			error = emptyApError(filterName, area, whtDat, self.pscale, self.survey) * photflam
			if np.isfinite(flux):
				self.photometry[filterName].append(flux)
				self.photometry[filterName+'_err'].append(error)
			else:
				self.photometry[filterName].append(np.nan)
				self.photometry[filterName+'_err'].append(np.nan)

	def write_FAST(self, filters, translate, images, num):
		'''This makes a list of strings nAnnuli long; each string is one
		full line for a FAST input file. 'Order' is the order that we should
		write the filters. This is provided because FAST isn't smart enough
        to re-build the libraries if you have the same filters but they're
        in a different order. '''
		
        # initialize the list of lines we'll want to write out
        fastList = []
        
        # make a string for each annulus
		for ann in range(self.nAnnuli):
            
			# start with the galaxy ID and redshift
			galStr = str(num) + str(ann) + '\t' + str(self.z) + '\t'
            
            # now, for each filter we need to write out both the flux and the error
            # in that filter
			for fil in filters:
                
				# if it's a filter where we've done resolved photometry...
				if fil in images:
					# make sure we've done calculation
					if fil in self.photometry.keys():
						# if it's not nan, write it out
						if np.isfinite(self.photometry[fil][ann]):
							fnu = Fnu25(self.photometry[fil][ann], translate[fil][0])
							fnu_err = Fnu25(self.photometry[fil+'_err'][ann], translate[fil][0])
							galStr += str(fnu) + '\t' + str(fnu_err) + '\t'
						# if it's nan, write -99s
						else:
							galStr+= '-99\t-99\t'
					else:
						galStr+='-99\t-99\t'
                        
				# for filters without resolved photometry, just write out empties
                # we do this so that FAST will predict the values in this filter,
                # which we'll use later to calculate the integral constraint
                # described in Wuyts+12
				else:
					galStr+= '-99\t-99\t'
			fastList.append(galStr)
		return fastList

	def calc_corr(self, images):
		'''calculate 'correction' that brings total flux down by factor of
		(mean of) diff b/t measured and catalog flux. This is really just
        an aperture correction that accounts for how far out we were able
        to measure aperture photometry.'''
        
        # initialize lists for the measured fluxes
		totMeas = []; totErr = []; divErr = []; catfl = []
        
        # for every filter where we have resolved photometry, find the
        # total flux we measured in that filter
		for im in images:
            # measured flux & error
			totMeas.append(np.sum(self.photometry[im])) # total flux
			totErr.append(np.sqrt(np.sum(np.array(self.photometry[im+'_err'])**2.))) # flux error
			
            # calculate the percentage error (used to weight each filter)
            divErr.append(totMeas[-1]/self.catalogFlux['f_'+im.lower()] * \
				np.sqrt((totErr[-1]/totMeas[-1])**2. +
				(self.catalogError['f_'+im.lower()] /
				self.catalogFlux['f_'+im.lower()])**2.))
			
            # record the catalog flux in that filter
            catfl.append(self.catalogFlux['f_'+im.lower()])
            
		# do weighted average for correction (making sure to leave out NaNs)
		self.corr = np.average(np.nan_to_num(np.array(totMeas)/np.array(catfl)),
			weights=np.nan_to_num(1/np.array(divErr)**2.))

	def get_Lg(self, images, translate):
        '''Get the flux in the *rest-frame* SDSS g band filter as a function
        of radius. We want to use EAZY to interpolate our measured ~5-8 bands
        to the rest-frame g band. This is to report M/L_g for comparison
        to literature values.'''
        
		# write out a catalog file for EAZY
		with open('EAZY/Lg.cat', 'w') as f:
			# write header
			f.write('# id\tz_spec ')
			
            # write out the 'translate' value for each measured resolved 
            # filter in header so EAZY knows where to find filter curves
            s = ''
			for im in images:
				s+=translate[im][1] + '\t' 
				s+='E'+translate[im][1][1:] + '\t'
			f.write(s+'\n')

			# write each annulus as a separate "galaxy"
			for i in range(self.nAnnuli):
                # id and redshift
				f.write(str(i) + '\t' + str(self.z) + '\t')
                
                # flux & error for each band of resolved photometry
                # make sure these are in zpt 25 fluxes (that's what
                # EAZY expects!)
				for im in images:
                    # make sure photometry was measured well
					if not np.max(np.isnan(self.photometry[im])):
						f.write('{:.5f}'.format(Fnu25(self.photometry[im][i],
							translate[im][0]))+'\t')
						f.write('{:.5f}'.format(Fnu25(self.photometry[im+'_err'][i],
							translate[im][0])) + '\t')
					# otherwise write out no-data-values
                    else:
						f.write('-99\t-99\t')
				f.write('\n')

		# now we've written the catalog, run EAZY on rest-frame g filter
		os.chdir("EAZY")
		return_code = subprocess.call("./eazy > 'logLg.log'", shell=True)
		os.chdir("..")

		# after running EAZY, read in EAZY results
        # save them so we have for later plotting
		g = np.loadtxt('EAZY/OUTPUT/Lg.157.rf')[:,5]
		self.Lg = flux2lum((g * 10**(-29.44)), 3e18/lamg, self.z)
		self.LgErr = flux2lum(self.photometry['F160W_err'], translate['F160W'][0], self.z)
		
		# also save L_F160W for fitting 
		self.LF160 = flux2lum(self.photometry['F160W'], translate['F160W'][0], self.z)
		self.LF160Err = flux2lum(self.photometry['F160W_err'], translate['F160W'][0], self.z)

	def read_fast(self, IDs, folder, fastOut, grid):
		''' Once we've run FAST to calculate the mass in each annulus, need to
        actually read in the FAST results! This reads in the chi and scale grid for 
        all annuli. We'll carry them around with the galaxy object while doing the
        analysis, but remove them before saving the file to reduce total size (they 
        are HUGE!) '''
		
        # get ID, z as it is in FAST
        # this mirrors the write_fast function above
		num = np.where(np.array(IDs)==self.id)[0][0]+1
		if num >= 10: # avoid confusion between 110 (1-10) and 110 (11-0)
			num = num*10
		zIdx = closest(self.z, grid['z'])

        # each annulus has a different chi and scale file; read each in
		for ann in range(self.nAnnuli):
			# read in files (stored as IDL .save files)
			chiFile = scipy.io.readsav(folder + '/chi_'+ fastOut[:-4] + '.' +
				str(num) + str(ann) + '.save')
			scaleFile = scipy.io.readsav(folder + '/scale_'+ fastOut[:-4] + '.'
				+ str(num) + str(ann) + '.save')
                
            # add the chi^2 grid to the galaxy object   
            # index it to get rid of the redundant metal and z index (there's only one thing) 
			self.chiGrid.append(chiFile['chi'][:,:,0,:,0]) 
            
            # add in the chi^2 threshhold
			self.chiThresh.append(chiFile['chi_thr'])
            
            # and the scale factor grid that gets from model -> data
			scale = scaleFile['scale'][:,:,0,:,0] # Av x age x tau
			self.massGrid.append(chiFile['mass'][:,:,0,:,0])

			# get best-fit position, error bars
			# best position is indices
			self.bestPos.append((chiFile['i_best'][4], chiFile['i_best'][3],
				chiFile['i_best'][1]))	# Av, logAge, logTau indices

			# scale the grid to this annulus, convert 25->AB magnitude
			self.grid.append(np.multiply(grid['grid'][..., zIdx],
				scale[None,...]) * (10.**(-29.428)))

		# because we'll change around position later, save the original positions
		self.fastPos = np.copy(self.bestPos)

	def get_chisq(self, posList, filterOrder):
		'''Get total chi squared value in some annulus at a given set of
		(Av, logAge, logTau) positions. posList is nAnnuli long list of
		tuples that gives (Av, logAge, logTau) for each annulus.
		filterOrder is the order that the filters exist in FAST'''

		# sum of FAST chi square for each annulus
		fastChi = np.sum([self.chiGrid[ann][posList[ann]]
			for ann in range(self.nAnnuli)])

		# get chi square for all integrated filters
		modelFlux = np.sum([self.grid[ann][(slice(None),)+posList[ann]]
			for ann in range(self.nAnnuli)], axis=0)
		catFlux = np.array([self.catalogFlux['f_'+i.lower()] for i in filterOrder])
		catErr = np.array([self.catalogError['f_'+i.lower()] for i in filterOrder])
		intChi = np.sum((catFlux - modelFlux*self.corr)**2. / catErr**2.)

		# reduce chi square and return it
		chi_red = 10*fastChi / (len(self.photometry.keys())/2 - len(posList[0])) + \
			intChi / (len(filterOrder) - len(posList[0]))
		return(chi_red)

	def adjustPos(self, positions, ann, filterOrder):
		'''Adjust position of one annulus (number ann) within +/- 3 steps to
		the new lowest chi-squared position. Positions is the list of annuli
		positions that we're starting from-- not having this just be bestPos so
		that it's easy to do either best-fit or MC error positions.
		filterOrder is the order that the FAST filters were written in.'''
		
        # make a list of the possible values our age, tau, Av can now
        # take on. Can move +/- 3 steps in any variable.
        newPos = [tuple(map(sum, zip(positions[ann], i)))
			for i in itertools.product([0,-1,1,-2,2,-3,3], repeat=3)]
            
        # initialize housekeeping variables    
		newChi = np.zeros(len(newPos))+1e10
		tmpPos = [i for i in positions]
		changed = False

        # for each possible new position, calculate the chi^2
		for posIdx, pos in enumerate(newPos):
			
            # make sure this position is actually allowed
            # (e.g., doesn't hit the edge of the grid)
			if (0 <= pos[0] < self.grid[0].shape[1]) \
				and (0 <= pos[1] < self.grid[0].shape[2]) \
				and (0 <= pos[2] < self.grid[0].shape[3]):

				# make sure that the values aren't NaN
				if np.amax(np.isnan(self.grid[ann][(slice(None),)+pos])):
					continue

				# calculate chi square for new position
				tmpPos[ann] = pos
				newChi[posIdx] = self.get_chisq(tmpPos, filterOrder)

		# get lowest chi square of new possible positions (if it didn't rail)
		if np.amin(newChi) < 1e10:
			# see if it changed
			if newPos[np.argmin(newChi)] != positions[ann]:
				changed = True
			# set new positions
			positions[ann] = newPos[np.argmin(newChi)]
        # and return    
		return (positions, changed)

	def find_bestPos(self, filterOrder):
		'''Starting from the FAST best-fit positions, find the optimized
		best fit positions for each annulus. This will run the above
        adjustPos function many times.'''
		
        # set the maximum number of iterations
        # (tested, and we don't ever actually hit this... still good to have)
        maxIter = 500
        
        # initialize counters for how many times we've updated; how
        # long it's been stable at a chi^2 minimum
		it, lastChanged = 0, 0
        
        # while we haven't converged, update the positions
		while (it < maxIter and lastChanged < self.nAnnuli*3):
			# find new positions
			ann = it % self.nAnnuli
			self.bestPos, changed = self.adjustPos(self.bestPos, ann, filterOrder)
			# if unchanged, increment counter
			if not changed:
				lastChanged += 1
			it += 1

	def find_posErrors(self, nsim, grid, filterOrder):
        '''Monte Carlo simulation to get errors in the best-fit
        age, tau, Av for each annulus. For each simulation, pick a 
        random starting value (w/i error) for each position. Then, run
        the same constraint process starting from those random positions. '''
        
		# as above, set max iters
		maxIter = 500
		
        # run nsim simulations
        simPos = []
		for sim in range(nsim):

			# pick random starting positions for each annulus
			pos = []
			for ann in range(self.nAnnuli):
				oneSig = np.where((self.chiGrid[ann] < self.chiThresh[ann] + 2) &
					(self.chiGrid[ann] > self.chiThresh[ann] - 2))
				randIdx = np.random.randint(0, len(oneSig[0]))
				pos.append((oneSig[0][randIdx], oneSig[1][randIdx], oneSig[2][randIdx]))

			# constrain (as above) but starting from Monte Carlo position
			it, lastChanged = 0, 0
			while (it < maxIter and lastChanged < self.nAnnuli*3):
				# find new positions
				ann = it % self.nAnnuli
				pos, changed = self.adjustPos(pos, ann, filterOrder)
				# if unchanged, increment counter
				if not changed:
					lastChanged += 1
				it += 1
			# and save for later
			simPos.append(pos)

		return simPos

	def get_masses(self, filterOrder, nsim, grid, images, translate):
        '''Putting it all together: get the stellar mass in each annulus.'''
               
		# first, set Lg
		self.get_Lg(images, translate)

		# find best position for each annulus
		self.find_bestPos(filterOrder)
        
        # use the massGrid to find the corresponding mass for that best-fit position
		self.bestMasses = np.array([self.massGrid[i][self.bestPos[i]] for i
			in range(self.nAnnuli)])
            
        # also record the M/L_g and M/L_f160    
		self.bestML = self.bestMasses / self.Lg
		self.bestML_f160 = self.bestMasses / self.LF160

		# then, do Monte Carlo simulations to find the errors in the masses
        # do the simulation to find the error in the position
		sims = self.find_posErrors(nsim, grid, filterOrder)
        # convert it to mass
		masses = np.array([np.array([self.massGrid[i][sim[i]] for i \
			in range(self.nAnnuli)]) for sim in sims])
        # find the upper/lower 68% confidence interval    
		upper_sig = np.clip(np.nanpercentile(masses, 84, axis=0) - self.bestMasses,
			0, np.inf)
		lower_sig = np.clip(self.bestMasses - np.nanpercentile(masses, 16, axis=0),
			0, np.inf)
		# enforce a minimum 10% error on masses
		upper_sig[upper_sig < 0.1*self.bestMasses] = \
			0.1*self.bestMasses[upper_sig < 0.1*self.bestMasses]
		lower_sig[lower_sig < 0.1*self.bestMasses] = \
			0.1*self.bestMasses[lower_sig < 0.1*self.bestMasses]
        # and record results    
		self.massErrors = np.array([lower_sig, upper_sig])

		# find M/L for all simulated best positions
        # will want to also perturb L to get a real M/L error
        # error in L_g
		Lgs = np.array([np.random.normal(loc=self.Lg[i], scale=self.LgErr[i], size=nsim)
			for i in range(len(self.Lg))]).transpose()            
        # error in L_f160
		Lgs_F160 = np.array([np.random.normal(loc=self.LF160[i], scale=self.LF160Err[i], size=nsim)
			for i in range(len(self.LF160))]).transpose()	 
        # get upper/lower confidence intervals for both g & F160 filters    
		upper_sig = np.clip(np.nanpercentile(masses/Lgs, 84, axis=0) - self.bestML,
			0, np.inf)
		lower_sig = np.clip(self.bestML - np.nanpercentile(masses/Lgs, 16, axis=0),
			0, np.inf)
		upper_sig_f160 = np.clip(np.nanpercentile(masses/Lgs_F160, 84, axis=0) - self.bestML_f160,
			0, np.inf)
		lower_sig_f160 = np.clip(self.bestML_f160 - np.nanpercentile(masses/Lgs_F160, 16, axis=0),
			0, np.inf)	  
		# ensure minimum 5% error on M/L
		upper_sig[upper_sig < 0.05*self.bestML] = \
			0.05*self.bestML[upper_sig < 0.05*self.bestML]
		lower_sig[lower_sig < 0.05*self.bestML] = \
			0.05*self.bestML[lower_sig < 0.05*self.bestML]
		self.MLerr = np.array([lower_sig, upper_sig])
		upper_sig_f160[upper_sig_f160 < 0.05*self.bestML_f160] = \
			0.05*self.bestML_f160[upper_sig_f160 < 0.05*self.bestML_f160]
		lower_sig_f160[lower_sig_f160 < 0.05*self.bestML_f160] = \
			0.05*self.bestML_f160[lower_sig_f160 < 0.05*self.bestML_f160]
		self.MLerr_f160 = np.array([lower_sig_f160, upper_sig_f160])
