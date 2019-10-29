'''The following code implements the Szomoru+11,12,13 method for measuring half-mass radii.
In brief, the method works as follows:
1) Use GALFIT to fit a sersic profile to each band of imaging
2) Calculate a 1D light profile in each band from the best-fit sersic parameters
2) Use aperture photometry to calculate the flux in the *residual* image from
    GALFIT as a function of radius. Add this residual flux back into the 1D
    light profile to get a more accurate / less model-dependent light profile
3) Use EAZY to interpolate between the observed light profiles to rest-frame
    u and g light profiles
4) Calculate the u-g color as a function of radius
5) Use the full galaxy sample to create a relation between u-g and M/L_g
6) Use this relation to turn our measured u-g profile into a M/L_g profile
7) Multiply by g-band luminosity profile to get a mass profile
8) Find the half-mass radius
9) Do Monte Carlo simulations to get error bars on the mass profile and the half-
    mass radius: vary the u-g color profile within its error bars, re-calcute
    M/L, M, and r_mass.
 '''


# import a few packages that we'll need
from photFuncs import *
import os
import subprocess
import scipy.io
import sys
from scipy.special import gammainc
from scipy.optimize import curve_fit
import glob
from writeGalfit import *

# import some plotting packages; set pretty defaults
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
plt.interactive(True)
sns.set(style="ticks", font_scale=1.5)
sns.set_style({"xtick.direction":u'in', "ytick.direction":u'in', "lines.linewidth":1.2})

##################################### FUNCTIONS #####################################

def sersicLum(mtot, r, re, n, lambFil):
    '''Returns the flux (F_lambda) interior to a given radius "r" for a 
    Sersic luminosity profile. Mtot is the total mass of the galaxy, re
    is the effective radius, n is the sersic index, and lambFil is the
    central wavelength of the filter we're looking at (in *Angstroms*) '''
    
    # use Graham & Driver approx for sersic index-- first check we're in valid range
	if (n < 0.36):
		print('Warning: n is outside bounds for analytic approximation!')
	b = 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2.) + 131/(1148175*n**3.) - \
        2194697/(30690717750*n**4)  
	
    # now find the flux at r<=r (again, eqn from Graham & Driver 2005)
    x = b * (r / re)**(1./n)
	return ((2.998e18/lambFil**2.) * 10**(-2./5 * (mtot+48.6)) * gammainc(2*n, x))

def fitfunc(x, slope, intercept):
    '''Later, we'll be fitting a straight line to a (u-g) - M/L relation--- make
    sure that we have a function describing a simple line to do the fit!'''
	return slope*x + intercept

##################################### SETUP #####################################
'''Get set up to run this method for modeling the intrinsic M/L! Want to load 
in all of the work we've already done as well as the survey
data we'll need to run this file. '''

# this program should be called command line as "python fit_method3.py group"
# where "group" is the name of the param file & savefile used in measure_ML.py
# if no group is listed, exit
if len(sys.argv) < 2:
	sys.exit('Please list the group as a command-line argument')
group = sys.argv[1]

# open up the master 3D-HST catalog
master = fits.open('/Volumes/DarkPhoenix/Surveys/3D-HST/3dhst.v4.1.5.master.fits')

# use function in photFuncs.py to parse in the input file and check to make sure
# that all the paths actually exist.
inputfile = 'input_files/'+group+'.param'
survey, IDs, zs, filterResPath, translatePath, galfitPath, catalogPath, images, \
     filters, rmFASTgrid, pixScale, psf, fastPath, fastDir, fastOut, library, \
     nsim, savedict, imPaths = readInputFile(inputfile)

# make sure we have a name to save files to...
savename = group

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

# open PDF for figures
pdf = PdfPages('Plots/szomoru_'+group+'.pdf')

# central wavelengths of SDSS u & g filters 
lam_u = 3.56179e+03
lam_g = 4.71887e+03

# make translate file
# keys are the names of the filters (as they appear in the 3D-HST survey); 
# values are the central wavelength of the filter and the EAZY/FAST filter number
translate = makeTranslateDict(filterResPath, translatePath)

# open up survey files we'll use later
galfit = np.loadtxt(galfitPath)
		#	3dhst_NUMBER, RA, DEC, f, mag, dmag, re, dre, n, dn, q, dq, pa, dpa, sn
cat = fits.open(catalogPath)
fast = fits.open(fastPath)
hduSeg = fits.open(imPaths['seg'])

# make dict to store results
# 'sz' will hold the fluxes that we measure
sz = {}
# 'sz_results' will hold all of the derived quantities (e.g., radii)
sz_results = {}
for galID in IDs:
	sz[galID] = {} # will have keys of filters, values array of fluxes
	sz_results[galID] = {} # keys: measured quantities ('re50', 're84', etc)

# make a (very basic) header for galfit runs
# (b/c supplying sigma image, only need exptime)
exptime = 1000.
header = fits.Header()
header['EXPTIME'] = exptime


##################################### RUN GALFIT #####################################
'''In this section, we will prepare cutouts in each filter where we have imaging.
Then, we'll run GALFIT on those cutouts and read in the results as 1D profiles. '''

# will want to make a cutout, run galfit for each band w/ resolved imaging
for filterName in images:
    
    ########################################################
    ''' First, get set up to make cutouts in this filter '''
    
    # start with an update for the user since this will take a bit...
	print('Started measuring in filter' + filterName)
	
    # open the data and weight images for this band of imaging
	hduData = fits.open(imPaths[filterName])
	wcsData = WCS(hduData[0].header)
	paData = 0.0 # from F160W
	photflam = hduData[0].header['photflam']
	hduWht = fits.open(imPaths[filterName+'_wht'])

	# make sure that all bands have a measured photflam (some in UDS are incorrectly
    # marked as 0...) 
	if photflam == 0:
		print('Using photflam from COSMOS for filter '+filterName)
		if filterName == 'F140W': photflam = 1.4737148e-20
		elif filterName == 'F160W': photflam = 1.9275602e-20
		elif filterName == 'F125W': photflam = 2.24834169999999e-20
		elif filterName == 'F606W': photflam = 7.8624958e-20
		elif filterName == 'F814W': photflam = 7.0331885e-20
		else: sys.exit('No photflam found.')

	# GALFIT requires the photometric zeropoint (for setting the sky value).
    # here, choose the appropriate zeropoint for this band/filter using
    # the photometric zeropoints listed in Skelton+
	if survey == 'cosmos':
		if filterName == 'F160W':
			zpt = 25.956
		elif filterName == 'F140W':
			zpt = 26.465
		elif filterName == 'F125W':
			zpt = 26.247
		elif filterName == 'F606W':
			zpt = 26.491
		elif filterName == 'F814W':
			zpt = 25.943
	elif survey == 'uds':
		if filterName == 'F160W':
			zpt = 26.452
		elif filterName == 'F140W':
			zpt = 26.452
		elif filterName == 'F125W':
			zpt = 26.230
		elif filterName == 'F606W':
			zpt = 26.491
		elif filterName == 'F814W':
			zpt =  25.943
	elif survey == 'cdfs':
		if filterName == 'F160W':
			zpt = 25.946
		elif filterName == 'F125W':
			zpt = 26.230
		elif filterName == 'F140W':
			zpt =  26.452
		elif filterName == 'F435W':
			zpt =  25.690
		elif filterName == 'F606W':
			zpt = 26.511
		elif filterName == 'F775W':
			zpt = 25.671
		elif filterName == 'F850LP':
			zpt = 24.871
		elif filterName == 'F814WCAND':
			zpt =  25.947
	else:
		print('Need to grab right zeropoints for this survey!')
		raise SystemExit

    ########################################################
    ''' Now we're set up, go make cutouts and run galfit! '''
    
    for galID in galDict.keys():
		
        # make cutouts of data, weight, and segmentation arrays for this galaxy
		pixY, pixX = galDict[galID].get_pix(wcsData)
		dat = cutout(hduData[0].data, pixX, pixY, pscale = galDict[galID].pscale)
		whtDat = cutout(hduWht[0].data, pixX, pixY, pscale = galDict[galID].pscale)
		segDat = cutout(hduSeg[0].data, pixX, pixY, pscale = galDict[galID].pscale)
		
        # mask out *other galaxies* (not sky) using SExtractor segmentation map
        mask = (segDat!=galDict[galID].id) & (segDat!=0) 

        # check to make sure that the array isn't simply full of NaN values.
        # this is true sometimes for F140W if that part of the sky didn't get any coverage
		if np.amin(np.isnan(dat)):
			print('array is NaN for galaxy '+str(galID)+' in filter '+filterName)
			continue
		
        # similarly, make sure there is actually non-zero data in the cutout...
        if np.nanmax(dat) == 0:
			print('array is 0 for galaxy '+str(galID)+' in filter '+filterName)
			continue	

        # galfit likes NaN values to be zero-- replace all NaNs with 0s
		dat[np.isnan(dat)] = 0
		whtDat[np.isnan(dat)] = 0

		# write out fits files for the data, weight, and mask arrays
        # galfit wants units of counts, so multiply by an arbitrary exptime
        # to get values on the right scale for galfit to not complain
		fits.writeto('galfitInputs/'+group+'_dat.fits',
			data=dat*exptime,
			header = header, overwrite=True)
        # make sure we're not dividing by 0 to make 1/wht^2 map...    
		fits.writeto('galfitInputs/'+group+'_wht.fits',
			data=np.sqrt(np.abs(dat*exptime) + np.divide(exptime**2., whtDat,
			out=np.zeros_like(whtDat), where=np.sqrt(whtDat)!=0)),
			header = header, overwrite=True)
		fits.writeto('galfitInputs/'+group+'_mask.fits',
			data=mask.astype(int),
			header = header, overwrite=True)
			
		# for the constraint file, we want the total integrated magnitude of the
        # galaxy in that filter (get from the master catalog)
        # as well as the expected location of the center of the galaxy	
		mag = 25 - 5/2. * np.log10(cat[1].data[galDict[galID].id - 1]['f_'+filterName])
		pos = [galDict[galID].pscale+pixY%1, galDict[galID].pscale+pixX%1]
		if np.isnan(mag):
			print('magnitude is NaN for galaxy '+str(galID)+' in filter '+filterName)
			continue

		# write out the galfit param file
        # as described in Suess+19, we fix the sky background to 0.
		galfit_write_param('galfitInputs/'+group+'.input', group+'_dat.fits', \
			group+'_output.fits', group+'_wht.fits', "psf_"+survey+".fits", \
			group+'_mask.fits', group+'.constraints', dat.shape, zpt, \
			str(galDict[galID].pscale+pixX%1)[:7], str(galDict[galID].pscale+pixY%1)[:7], str(mag)[:7], \
			str(galDict[galID].re/pixScale)[:7], str(galDict[galID].n)[:7], str(galDict[galID].q)[:7], \
			str(galDict[galID].pa)[:7], sky='0.0')		

		# write out the galfit constraints file
        # the constraints are described in detail in Suess+19 (but are fairly standard)
		galfit_write_constraints('galfitInputs/'+group+'.constraints', mag, \
			nmin='0.2', nmax='8.0', remin='0.3', remax='400')
			
		# actually run galfit!
		os.chdir("galfitInputs")
		return_code = subprocess.call('/Users/wren/galfit/galfit '+
			group+'.input > '+group+'.log', shell=True)
		os.chdir("..")

        ########################################################
        ''' Read in galfit results '''

		# first, make sure that galfit actually produced an output file
		if not os.path.isfile('galfitInputs/'+group+'_output.fits'):
			print('No output file found for galaxy '+str(galID)+' in filter '+filterName)
			continue
        # if so, open the output file    
		hduOut = fits.open('galfitInputs/'+group+'_output.fits')
		
		# check output flags-- if galfit crashed, move on
		if ('2' in hduOut[2].header['flags'].split(' ')) or \
			('1' in hduOut[2].header['flags'].split(' ')):
			print('Bad galfit flag for galaxy '+str(galID)+' in filter '+filterName)
			continue
			
        # very, veeeery occasionally (~1/10,000x) galfit will seg fault.
        # in this case, it returns a default output file. We can tell this default
        # file apart from a real output file easily b/c it is a different size
        # than the cutouts we've made.
        # if this happened, move on. 
		if hduOut[3].shape != dat.shape:
			print('Seg fault for galaxy '+str(galID)+' in filter '+filterName)
			continue	

        # usually, galfit will produce both a value and an error
        # with some types of errors, it won't give us an error value
        # this isn't a big deal since we're not going to trust their
        # error bars anyways (likely underestimated) but make sure we
        # account for this possibility
        
        # read in the total magnitude
		if len(hduOut[2].header['1_mag'].split()) == 3:
			mag = float(hduOut[2].header['1_mag'].split()[0])
			magErr = float(hduOut[2].header['1_mag'].split()[2])
		else:
			mag = float(hduOut[2].header['1_mag'].split()[0][1:-1])
			mag = np.nan
		
        # read in the half-light radius
        try:
			re = float(hduOut[2].header['1_re'].split()[0])
			reErr = float(hduOut[2].header['1_re'].split()[2])
		except ValueError:
			re = float(hduOut[2].header['1_re'].split()[0][1:-1])
			reErr = np.nan
		
        # read in the sersic index
        try:
			n = float(hduOut[2].header['1_n'].split()[0])
			nErr = float(hduOut[2].header['1_n'].split()[2])
		except ValueError:
			n = float(hduOut[2].header['1_n'].split()[0][1:-1])
			nErr = np.nan
		
        # read in the central x location
        if len(hduOut[2].header['1_xc'].split()) == 3:
			xc = float(hduOut[2].header['1_xc'].split()[0])
			xcErr = float(hduOut[2].header['1_xc'].split()[2])
		
        # read in the central y location
        if len(hduOut[2].header['1_yc'].split()) == 3:
			yc = float(hduOut[2].header['1_yc'].split()[0])
			ycErr = float(hduOut[2].header['1_yc'].split()[2])
		
        # read in the axis ratio q
        try:
			q = float(hduOut[2].header['1_ar'].split()[0])
			qErr = float(hduOut[2].header['1_ar'].split()[2])
		except ValueError:
			q = float(hduOut[2].header['1_ar'].split()[0][1:-1])
			qErr = np.nan
		
        # read in the position angle
        try:
			pa = float(hduOut[2].header['1_pa'].split()[0])
			paErr = float(hduOut[2].header['1_pa'].split()[2])
		except ValueError:
			pa = float(hduOut[2].header['1_pa'].split()[0][1:-1])
			paErr = np.nan

        # ok, now we have all of the derived sersic quantities from the galfit fit,
		# construct an intrinsic sersic profile out to 10kpc for this galaxy
		re_kpc = re*pixScale / cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value
		
        # set the minimum r we'll look at to 1/4 pixel size (somewhat arbitrary)
		min_r = pixScale / 1./4 / cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value # set min at 1/4 pixel
		
        # make an array of rs (in kpc) to evaluate profile
        # following Szomoru+, we will correct the first 10kpc with the measured residual
        # (the rest of the profile will just be the uncorrected sersic profile)
        rs_kpc = np.logspace(np.log10(min_r), 2.0, 100) 
		
        # how do those kpc-valued rs map onto pixels so we can actually do measurement?
		rs_pix = rs_kpc / pixScale * cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value
		
        # calculate the flux within each annulus (r 0-1, 1-2, 2-3, ...)
		Ls = sersicLum(mag, rs_pix, re, n, translate[filterName][0])[1:] - \
			sersicLum(mag, rs_pix, re, n, translate[filterName][0])[:-1]
        
        ########################################################
        ''' Measure the flux in the galfit residual image '''    
        # we'll want to do this similarly to how we measured aperture photometry
        # in measure_ML.py
        # need to keep track of the apertures as well as their fluxes, errors, and areas  
		aps = []
		flux = []
		error = []  
		areas = np.zeros(len(rs_pix))
        
        # go ahead and make the first aperture have radius r=0
        # (so that the first real "annulus" is actually just a circle)
		apIn = photutils.EllipticalAperture(positions=(xc-1, yc-1), a=rs_pix[0],
			b=rs_pix[0]*q, theta=pa+45)

		# actually measure the flux in the residual image w/ aperture photometry
		for i, r in enumerate(rs_pix[1:]):
            # the below is directly taken from measure_ML.py
			aps.append(apIn)
			apOut = photutils.EllipticalAperture(positions=(xc-1, yc-1),a=r,
				b=r*q, theta=pa+45) # + 90deg is to account for diff definitions of theta in photutils/galfit
			areas[i] = apOut.area() - apIn.area()
			maskArea = (photutils.aperture_photometry(mask, apOut)['aperture_sum'][0] -
				photutils.aperture_photometry(mask, apIn)['aperture_sum'][0])
			corr = areas[i] / (areas[i] - maskArea)
			error.append(emptyApError(filterName, areas[i], whtDat, len(dat)//2, survey) * photflam)

            # following Szomoru+, we *don't* add the residual back in if we're 
            # more than 10kpc from the center of the galaxy.
			if rs_kpc[i] > 10.0:
				flux.append(flux[-1])
				apIn = apOut
				continue

			# galfit image is in units of e-*1000, so multiply by photflam/1000 
            # to get results in Flambda
			flux.append((photutils.aperture_photometry(hduOut[3].data, apOut, mask=mask)['aperture_sum'][0] -
				photutils.aperture_photometry(hduOut[3].data, apIn, mask=mask)['aperture_sum'][0]) \
				* corr * photflam / exptime)

			# next inner aperture is the outer aperture for this annulus
			apIn = apOut

		# keep track of r and area for plotting
		rs_pix = rs_pix[1:]
		rs_kpc = rs_kpc[1:]
		areas = areas[:-1]

		# store for later
		sz[galID][filterName] = Ls + np.array(flux) # in Flambda
		sz[galID][filterName+'_err'] = np.array(error) # in Flambda
		sz[galID][filterName+'_area'] = areas # in pix^2
		sz[galID][filterName+'_r'] = rs_pix # in pix

# update the user		
print('Finished galfit runs')

# remove all of the galfit log files (we won't ever look at them, and there are a bunch...)
for f in os.listdir('galfitInputs/'):
	if f.startswith('galfit'):
		os.remove('galfitInputs/'+f)


################################## CALCULATE MASS PROFILES ##################################
''' Using the observed-frame light profiles we created above, derive the mass profile and 
half-mass radius for each galaxy. To do this, we need to:
1) run EAZY to get *rest-frame* u and g profiles
2) subtract these profiles to get a u-g color profile
3) create a mapping between u-g color and M/L using all galaxies in 3D-HST at similar redshifts
4) map our u-g color profile to a M/L profile
5) multiply by the L profile to get a mass profile
6) find the half-mass radius
7) Monte Carlo to get error bars
 '''

# for each galaxy in our sample, ...
for galID in galDict.keys():
	
    # first, we want to make sure that we can actually run EAZY.
    # check that we were able to measure data in at least one filter
	goodFils = np.intersect1d(list(sz[galID].keys()), images)
	if len(goodFils) == 0:
		print('Not sufficient information to run EAZY for galaxy '+str(galID))
		sz_results[galID]['M'] = np.nan
		sz_results[galID]['M_err'] = np.nan
		sz_results[galID]['ML'] = np.nan
		sz_results[galID]['ML_err'] = np.nan
		sz_results[galID]['ML_f160'] = np.nan
		sz_results[galID]['ML_f160_err'] = np.nan
		sz_results[galID]['Lg'] = np.nan
		sz_results[galID]['re'] = np.nan
		sz_results[galID]['re_err'] = np.nan
		sz_results[galID]['rs_pix'] = np.nan
		sz_results[galID]['rs_kpc'] = np.nan
		galDict[galID].re_szomoru = np.array((np.nan, np.nan)) 
		continue
	# and then make sure that we have measurements in at least one filter blueward of the RF u filter,
    # and one filter redward of the RF g filter (otherwise, we're *extrapolating* instead of interpolating.)
    if not (np.amin([translate[i][0] for i in goodFils]) < (lam_u * (1.+galDict[galID].z))) \
		& (np.amax([translate[i][0] for i in goodFils]) > (lam_g * (1.+galDict[galID].z))):
		print('Not sufficient information to run EAZY for galaxy '+str(galID))
		sz_results[galID]['M'] = np.nan
		sz_results[galID]['M_err'] = np.nan
		sz_results[galID]['ML'] = np.nan
		sz_results[galID]['ML_err'] = np.nan
		sz_results[galID]['ML_f160'] = np.nan
		sz_results[galID]['ML_f160_err'] = np.nan
		sz_results[galID]['Lg'] = np.nan
		sz_results[galID]['re'] = np.nan
		sz_results[galID]['re_err'] = np.nan
		sz_results[galID]['rs_pix'] = np.nan
		sz_results[galID]['rs_kpc'] = np.nan
		galDict[galID].re_szomoru = np.array((np.nan, np.nan, np.nan)) 
		continue

    # if we've gotten to this point, we have sufficient info to run EAZY
    # start by writing the .cat file that EAZY takes as input
	print('Writing .cat file for galaxy '+str(galID))
	with open('szomoru_files/'+group+'.cat', 'w') as f:
		# write file header
		f.write('# id\tz_spec ')
		# header has to have the translate #s for each filter w/ measured photometry
        s = ''
		for im in images:
			s+=translate[im][1] + '\t'
			s+='E'+translate[im][1][1:] + '\t'
		f.write(s+'\n')

		# write flux in each annulus as a separate "galaxy"
		for i in range(len(rs_kpc)):
			f.write(str(i) + '\t' + str(galDict[galID].z) + '\t')
			# if we have a light profile in this band, write out photometry
            for im in images:
				if im in sz[galID]:
					f.write('{:.5f}'.format(Fnu_arb_zpt(sz[galID][im][i],
						translate[im][0], zpt=32))+'\t')
					f.write('{:.5f}'.format(Fnu_arb_zpt(sz[galID][im+'_err'][i],
						translate[im][0], zpt=32)) + '\t')
				# otherwise, EAZY takes -99s as NaNs
                else:
					f.write('-99\t-99\t')
			f.write('\n')

	# edit the zphot.param file that EAZY takes as input to have it look at
    # the correct catalog, and to output to the correct file
    # note: we want to run first on one RF filter then the second 
    #   (allows EAZY to choose best template for each filter)
	with open('szomoru_files/zphot.param', 'r') as f:
		zphot = f.readlines()
	zphot[27] = 'CATALOG_FILE		  '+group+'.cat		 # Catalog data file\n'
	zphot[33] = 'MAIN_OUTPUT_FILE	  '+group+'	  # Main output file, .zout\n'
	zphot[60] = 'REST_FILTERS		  156\n' # first do u
	# write out the edited EAZY param file
    with open('szomoru_files/zphot.param', 'w') as f:
		f.writelines(zphot)

	# run EAZY on first rf filter
	os.chdir("szomoru_files")
	return_code = subprocess.call("./eazy > 'logs/"+group+"_156.log'", shell=True)
	os.chdir("..")

	# change the param file to calculate flux in the other RF filter
	zphot[60] = 'REST_FILTERS		  157\n' # then do g
	with open('szomoru_files/zphot.param', 'w') as f:
		f.writelines(zphot)

	# run EAZY on second rf filter
	os.chdir("szomoru_files")
	return_code = subprocess.call("./eazy > 'logs/"+group+"_157.log'", shell=True)
	os.chdir("..")

	# read in EAZY results
	u = np.nan_to_num(np.loadtxt('szomoru_files/OUTPUT/'+group+'.156.rf')[:,5])
	g = np.nan_to_num(np.loadtxt('szomoru_files/OUTPUT/'+group+'.157.rf')[:,5])
	# first choice is F160, but some galaxies have F140 or F125 instead
    # so try in decreasing order...
    try:
		u_err_mag = 2.5/np.log(10.) * sz[galID]['F160W_err'] / np.abs((3e18/lam_u**2. * u * 10**(-32.32)))
		g_err_mag = 2.5/np.log(10.) * sz[galID]['F160W_err'] / np.abs((3e18/lam_g**2. * g * 10**(-32.32)))
		lum_g_err = flux2lum(sz[galID]['F160W_err'], lam_g, galDict[galID].z)
		lum_f160 = flux2lum(sz[galID]['F160W'], translate['F160W'][0], galDict[galID].z)
		lum_f160_err = flux2lum(sz[galID]['F160W_err'], translate['F160W'][0], galDict[galID].z)
	except KeyError:
		try:
			u_err_mag = 2.5/np.log(10.) * sz[galID]['F140W_err'] / np.abs((3e18/lam_u**2. * u * 10**(-32.32)))
			g_err_mag = 2.5/np.log(10.) * sz[galID]['F140W_err'] / np.abs((3e18/lam_g**2. * g * 10**(-32.32)))
			lum_g_err = flux2lum(sz[galID]['F140W_err'], lam_g, galDict[galID].z)
			lum_f160 = np.nan
			lum_f160_err = np.nan
		except KeyError:
			u_err_mag = 2.5/np.log(10.) * sz[galID]['F125W_err'] / np.abs((3e18/lam_u**2. * u * 10**(-32.32)))
			g_err_mag = 2.5/np.log(10.) * sz[galID]['F125W_err'] / np.abs((3e18/lam_g**2. * g * 10**(-32.32)))	
			lum_g_err = flux2lum(sz[galID]['F125W_err'], lam_g, galDict[galID].z)
			lum_f160 = np.nan
			lum_f160_err = np.nan 
	
    # calculate g-band luminosity (for turning M/L_g -> M)
    lum_g = flux2lum((g * 10**(-32.32)), 3e18/lam_g, galDict[galID].z)
	
    # calculate u-g color and error on that color
    ug_color = -2.5 * np.log10(u / g)
	ug_color_err = np.sqrt(u_err_mag**2. + g_err_mag**2.)

	# when error in color is >0.2dex, set color value to threshold
    # prevents points with huge error bars from biasing results
    # see Szomoru+ for details
	ug_color_corr = ug_color
	ug_color_corr[ug_color_err > 0.2] = ug_color[ug_color_err <= 0.2][-1]

    # now, create a mapping from u-g color to M/L
    # do this by fitting a line between u-g and M/L for all 3D-HST
    # galaxies at the same redshift.
	
    # first, pick galaxies within 0.2 of our target redshift
    good = np.where((master[1].data['z_best'] > galDict[galID].z - 0.2) &
		(master[1].data['z_best'] < galDict[galID].z + 0.2) &
		(master[1].data['use_phot'] == 1) & (master[1].data['z_best_s'] < 3))
	
    # want to make sure we have a good number of galaxies-- if not, 
    # increase the redshift range we're looking over
    if not len(good[0]) > 500:
		good = np.where((master[1].data['z_best'] > galDict[galID].z - 0.5) &
			(master[1].data['z_best'] < galDict[galID].z + 0.5) &
			(master[1].data['use_phot'] == 1) & (master[1].data['z_best_s'] < 3))
		print('Used wider redshift range for galaxy ' +str(galID))
	
    # pull the rest-frame u and g F_lambda values from the master catalog
    all_flam_u = Flam((master[1].data['L156'][good] * (10.**(-6.44))), lam_u)
	all_flam_g = Flam((master[1].data['L157'][good] * (10.**(-6.44))), lam_g)
	all_lum_g = flux2lum(all_flam_g, lam_g, master[1].data['z_best'][good])
	
    # calculate the u-g color and the M/L
    all_ug_color = -2.5*np.log10(master[1].data['L156'][good] / master[1].data['L157'][good])
	all_ML = (10**master[1].data['lmass'][good]) / all_lum_g
	
    # fit a line to u-g vs M/L
    slope, intercept = curve_fit(fitfunc, all_ug_color, np.log10(all_ML))[0]
	# quick/easy way to get approx of scatter around the fit
	MLscatter = np.std(np.log10(all_ML) - fitfunc(all_ug_color, slope, intercept))

    # use this line to convert our measured u-g color to a M/L profile
	ML = 10**fitfunc(ug_color_corr, slope, intercept)
	
    # multiply by our RF g-band luminosity to get mass profile
    M = ML * lum_g
	
    # calculate M/L_F160W (to compare with the other measurement technique)
    ML_f160 = M / lum_f160
	
    # find the half-mass radius
    re = np.interp(np.sum(M)/2., np.cumsum(M), rs_kpc)

	
    #### Now, do Monte Carlo simulations to get error bars
    # we want to vary the u-g color w/i its uncertainties, re-calc M/L -> M -> r_mass
    # start by initializing some lists to hold the simulated values
	re_sim = []
	ML_sim = []
	ML_f160_sim = []
	M_sim = []
    
    # do a reasonable number of monte carlo draws
	for sim in range(1000):
		# calculate a new color profile by perturbing w/i error bars
        sim_color = np.random.normal(loc=ug_color_corr, scale = ug_color_err)
		sim_color[ug_color_err > 0.2] = sim_color[ug_color_err <= 0.2][-1]
		
        # we can use the same best-fit relation b/t u-g and M/L to get
        # the M/L profile (the galaxy is at the same redshift)
        sim_ML = 10**(fitfunc(sim_color, slope, intercept) + \
			np.random.normal(0, MLscatter, size=sim_color.shape))
		
        # calculate M, M/L_F160, r_mass; store results
        sim_M = sim_ML * np.random.normal(lum_g, lum_g_err, size=sim_color.shape)
		ML_f160_sim.append(sim_M/np.random.normal(lum_f160, lum_f160_err,size=sim_color.shape))
		re_sim.append(np.interp(np.sum(sim_M)/2., np.cumsum(sim_M), rs_kpc))
		ML_sim.append(sim_ML)
		M_sim.append(sim_M)
	
    # find 1sigma confidence interval
    re_err = np.array((np.nanpercentile(re_sim, 16), np.nanpercentile(re_sim,84))) # np.std(re_sim)
	ML_err = np.std(ML_sim, axis=0)
	ML_f160_err = np.std(ML_f160_sim, axis=0)
	M_err = np.std(M_sim, axis=0)
		
	# store all of our results for later
	sz_results[galID]['M'] = M
	sz_results[galID]['M_err'] = M_err
	sz_results[galID]['ML'] = ML
	sz_results[galID]['ML_err'] = ML_err
	sz_results[galID]['ML_f160'] = ML_f160
	sz_results[galID]['ML_f160_err'] = ML_f160_err
	sz_results[galID]['Lg'] = lum_g
	sz_results[galID]['re'] = re
	sz_results[galID]['re_err'] = re_err
    
    # for fun, also store the half-light radius in the rest-frame u and g bands
    # we can compare this to our other measurements as a consistency check
	flam_u = u * 3e18/(lam_u**2.) * 10**(-32.24)
	flam_g = g * 3e18/(lam_g**2.) * 10**(-32.24)
	sz_results[galID]['uhalf'] = np.interp(np.sum(flam_u)/2., np.cumsum(flam_u), rs_kpc)
	sz_results[galID]['ghalf'] = np.interp(np.sum(flam_g)/2., np.cumsum(flam_g), rs_kpc)
	
    # store the array of rs we used to calculate all of the profiles above
    # make sure we put it into a filter that actually has measurements...
    try:
		sz_results[galID]['rs_pix'] = sz[galID]['F160W_r']
	except KeyError:
		try:
			sz_results[galID]['rs_pix'] = sz[galID]['F140W_r']
		except KeyError:
			sz_results[galID]['rs_pix'] = sz[galID]['F125W_r']	  
	
    # for plotting, store the radius array in kpc as well
    sz_results[galID]['rs_kpc'] = rs_kpc
	
    # store the half-mass radius *in kpc* for later
	galDict[galID].re_szomoru = np.array((sz_results[galID]['re'], *sz_results[galID]['re_err'])) # in kpc!!!
	
    # also store the total mass (and error) that we derive
    # useful to compare this to the mass we get from fitting integrated photometry
    galDict[galID].M_szomoru = np.array((np.sum(sz_results[galID]['M']), 
        np.sqrt(np.sum(sz_results[galID]['M_err']**2.)))) # Mtot and error

	#### Diagnostic Plot ####
    
	# plot surface brightness profiles in the "best" band
    # (F160W, F140W, F125W)
	plt.figure(figsize=(8,8))
	plt.subplot(411)
	for f in images:
		if not f in sz[galID].keys(): continue
		mag_ab = -2.5*np.log10(sz[galID][f]*translate[f][0]**2./2.998e18) - 48.6
		area_arcsec = sz[galID][f+'_area'] * pixScale**2.
		mag_err = 2.5/np.log(10.) * sz[galID][f+'_err'] / np.abs(sz[galID][f])
		surfbright = mag_ab + 2.5*np.log10(area_arcsec)
		surfbright_err = mag_err
		plt.errorbar(rs_kpc, surfbright, yerr=surfbright_err, fmt='o', label=f)
	try:
		plt.errorbar(rs_kpc, 32-2.5*np.log10(u)+2.5*np.log10(sz[galID]['F160W_area'] * pixScale**2.),
			yerr = u_err_mag, label='u', fmt='o')
		plt.errorbar(rs_kpc, 32-2.5*np.log10(g)+2.5*np.log10(sz[galID]['F160W_area'] * pixScale**2.),
			yerr = g_err_mag, label='g', color='green', fmt='o')
	except KeyError:
		try:
			plt.errorbar(rs_kpc, 32-2.5*np.log10(u)+2.5*np.log10(sz[galID]['F140W_area'] * pixScale**2.),
				yerr = u_err_mag, label='u', fmt='o')
			plt.errorbar(rs_kpc, 32-2.5*np.log10(g)+2.5*np.log10(sz[galID]['F140W_area'] * pixScale**2.),
				yerr = g_err_mag, label='g', color='green', fmt='o')
		except KeyError:
			plt.errorbar(rs_kpc, 32-2.5*np.log10(u)+2.5*np.log10(sz[galID]['F125W_area'] * pixScale**2.),
				yerr = u_err_mag, label='u', fmt='o')
			plt.errorbar(rs_kpc, 32-2.5*np.log10(g)+2.5*np.log10(sz[galID]['F125W_area'] * pixScale**2.),
				yerr = g_err_mag, label='g', color='green', fmt='o')		 
	plt.gca().set_xscale('log')
	plt.ylim((30, 17))
	plt.legend(loc='best')
	plt.ylabel('mag / arcsec^2')
	plt.title(galID)
	plt.gca().tick_params(labelbottom='off')

	# plot u-g color and its error bars
	plt.subplot(412)
	plt.errorbar(rs_kpc, ug_color, yerr=ug_color_err, color='black')
	plt.semilogx(rs_kpc, ug_color_corr, color='red')
	plt.ylabel('u-g')
	plt.ylim((0,2.5))
	plt.gca().tick_params(labelbottom='off')
	
	# plot M/Lg profile
	plt.subplot(413)
	plt.errorbar(rs_kpc, ML, yerr=ML_err, color='black')
	plt.gca().set_xscale('log')
	plt.gca().set_yscale('log')
	plt.ylabel('M/Lg')
	plt.ylim((1e-1, 1e1))
	plt.gca().tick_params(labelbottom='off')
	
	# plot mass profile
	plt.subplot(414)
	plt.errorbar(rs_kpc, np.cumsum(M), yerr = np.sqrt(np.cumsum(M_err**2.)), color='black')
	plt.gca().set_xscale('log')
	plt.ylabel('M(<R)')
	plt.xlabel('r (kpc)')
    # mark where the half-mass radius is
	plt.axvline(re, label='re = '+str(re)[:5]+'+/-'+str(np.mean(re_err)-re)[:5], color='red')
    # mark the mass inferred from modeling photometry of the whole galaxy
    # (hopefully this is in good agreement with the total szomoru method mass...)
	plt.axhline(10**(fast[1].data['lmass'][np.where(fast[1].data['id'] == galID)[0][0]]), 
        color='grey', alpha=.6, ls='dashed')
	plt.legend(loc='best')
	
    # save the figure for this galaxy
    pdf.savefig()
	plt.close()
pdf.close()

# save all of our szomoru method results
np.savez('szomoru_files/'+savename+'.npz', sz_results=sz_results)

# since we updated galDict with .re_szomoru values, also re-save this file
np.savez('savefiles/'+savedict+'.npz', galDict=galDict) # has .re_szomoru [0]=re, [1]=re_err in kpc
