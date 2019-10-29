from photFuncs import *
import os
import subprocess
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy.special import gammainc
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import glob
from writeGalfit import *

plt.interactive(True)
sns.set(style="ticks", font_scale=1.5)
sns.set_style({"xtick.direction":u'in', "ytick.direction":u'in', "lines.linewidth":1.2})

def sersicLum(mtot, r, re, n, lambFil):
	# returns F_lambda interior to a given radius r for a Sersic profile.
	# lambFil should be in Angstroms
	if (n < 0.36):
		print('Warning: n is outside bounds for analytic approximation!')
	b = 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2.) + 131/(1148175*n**3.) - 2194697/(30690717750*n**4)  #1.9992*n - 0.3271
	x = b * (r / re)**(1./n)
	return ((2.998e18/lambFil**2.) * 10**(-2./5 * (mtot+48.6)) * gammainc(2*n, x))

def fitfunc(x, slope, intercept):
	# this is just a line
	return slope*x + intercept

# group should be listed as a command-line argument
if len(sys.argv) < 2:
	sys.exit('Please list the group as a command-line argument')
group = sys.argv[1]


master = fits.open('/Volumes/DarkPhoenix/Surveys/3D-HST/3dhst.v4.1.5.master.fits')
inputfile = 'input_files/'+group+'.param'
savename = group

try:
	galDict = np.load('savefiles/'+group+".npz", fix_imports = True, encoding = 'latin1')['galDict'][()]
except FileNotFoundError: # means I had to split dictionary up into separate savefiles
	files = glob.glob("savefiles/"+group+'*.npz')
	galDict = {}	
	for f in files:
		galDict = {**galDict, **np.load(f)['galDict'][()]}

# open PDF for figures
pdf = PdfPages('Plots/szomoru_'+group+'.pdf')

lam_u = 3.56179e+03
lam_g = 4.71887e+03

with open(inputfile) as f:
	lines = f.readlines()
lines = [i.rstrip().split(' = ') for i in lines if (i[0] != '#') & (i[:2] != '\n')] # strip out comments, etc

# start parsing out this file...
try: survey = [i for i in lines if i[0] == 'survey'][0][1]
except: sys.exit('Cannot read survey')

try: IDs = [int(j) for j in ([i for i in lines if i[0] == 'IDs'][0][1]).split(', ')]
except: sys.exit('Cannot read IDs')

try: filterResPath = [i for i in lines if i[0] == 'filterResPath'][0][1]
except: sys.exit('Cannot read filter res path')
if not os.path.isfile(filterResPath):
	sys.exit('No file for filter res at '+filterResPath)

try: translatePath = [i for i in lines if i[0] == 'translatePath'][0][1]
except: sys.exit('Cannot read translate path')
if not os.path.isfile(translatePath):
	sys.exit('No file for translate at '+translatePath)

try: galfitPath = [i for i in lines if i[0] == 'galfitPath'][0][1]
except: sys.exit('Cannot read galfit path')
if not os.path.isfile(galfitPath):
	sys.exit('No file for galfit at '+galfitPath)

try: catalogPath = [i for i in lines if i[0] == 'catalogPath'][0][1]
except: sys.exit('Cannot read catalog path')
if not os.path.isfile(catalogPath):
	sys.exit('No file for catalog at '+catalogPath)

try:
	images = [i for i in lines if i[0] == 'images'][0][1].split(' #')[0].split(', ')
	detectionBand = images[0]
except: sys.exit('Cannot read images')

try: pixScale = float([i for i in lines if i[0] == 'pixScale'][0][1])
except: sys.exit('Cannot read pixel scale')

try: psf = float([i for i in lines if i[0] == 'psf'][0][1].split(' #')[0])
except: sys.exit('Cannot read psf scale')

try: fastPath = [i for i in lines if i[0] == 'fastPath'][0][1]
except: sys.exit('Cannot read FAST path')
if not os.path.isfile(fastPath):
	sys.exit('No file for FAST at '+fastPath)

try: fastDir = [i for i in lines if i[0] == 'fastDir'][0][1]
except: sys.exit('Cannot read FAST directory')
if not os.path.isdir(fastDir):
	sys.exit('No FAST directory at '+fastDir)

try: fastOut = [i for i in lines if i[0] == 'fastOut'][0][1]
except: sys.exit('Cannot read FAST output directory')

try: library = [i for i in lines if i[0] == 'library'][0][1][:4]
except: sys.exit('Cannot read library')

try: nsim = int([i for i in lines if i[0] == 'nsim'][0][1])
except: sys.exit('Cannot read nsim')

try: savedict = [i for i in lines if i[0] == 'savedict'][0][1]
except: sys.exit('Cannot read dictionary save name')

imPaths = {}
try: imPaths['seg'] = [i for i in lines if i[0] == 'path_seg'][0][1]
except: sys.exit('Cannot read segmentation path')

for fil in images:
	try:
		imPaths[fil] = [i for i in lines if i[0] == 'path_'+fil][0][1]
		imPaths[fil+'_wht'] = [i for i in lines if i[0] == 'path_'+fil+'_wht'][0][1]
	except:
		sys.exit('Cannot read image or wht path for '+fil)
	if not (os.path.isfile(imPaths[fil]) and os.path.isfile(imPaths[fil+'_wht'])):
		sys.exit('No file found for filter '+fil)

#### Make translate file  ####
# generate list of filter lambda_c
fil = np.genfromtxt(filterResPath, usecols=1)
row = np.where(np.isnan(fil)==True)[0]
filList = []
with open(filterResPath) as fd:
	for n, line in enumerate(fd):
		if n in row:
			filList.append(line.split())

# open translate file and make dictionary
transFile = np.genfromtxt(translatePath, dtype='str')
translate = {}
for i in range(len(transFile))[::2]:
	row = filList[int(transFile[i][1][1:])-1]
	translate[transFile[i][0][2:].upper()] = [float(row[row.index('lambda_c=')+1]), \
		transFile[i][1]]

#### Grab files	 ####
# open files (have these be calls to some dictionary with filepaths...)
galfit = np.loadtxt(galfitPath)
		#	3dhst_NUMBER, RA, DEC, f, mag, dmag, re, dre, n, dn, q, dq, pa, dpa, sn
cat = fits.open(catalogPath)
fast = fits.open(fastPath)

#### make dict to store values ####
sz = {}
sz_results = {}
for galID in IDs:
	sz[galID] = {} # will have keys of filters, values array of fluxes
	sz_results[galID] = {}

#### Make cutout and weight images for galfit inputs ####
hduSeg = fits.open(imPaths['seg'])

# make (very basic) header for galfit runs
# (b/c supplying sigma image, only need exptime)
exptime = 1000.
header = fits.Header()
header['EXPTIME'] = exptime

#### Prepare & run galfit, read in results ####
for filterName in images:
	print('Started measuring in filter' + filterName)
	# open images
	hduData = fits.open(imPaths[filterName])
	wcsData = WCS(hduData[0].header)
	paData = 0.0 # from F160W...
	photflam = hduData[0].header['photflam']
	hduWht = fits.open(imPaths[filterName+'_wht'])

	# UDS some photflam are zero-- fix this!
	if photflam == 0:
		print('Using photflam from COSMOS for filter '+filterName)
		if filterName == 'F140W': photflam = 1.4737148e-20
		elif filterName == 'F160W': photflam = 1.9275602e-20
		elif filterName == 'F125W': photflam = 2.24834169999999e-20
		elif filterName == 'F606W': photflam = 7.8624958e-20
		elif filterName == 'F814W': photflam = 7.0331885e-20
		else: sys.exit('No photflam found.')

	# photometric zeropoints from Ros' paper:
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

	for galID in galDict.keys():
		# make cutouts of data
		pixY, pixX = galDict[galID].get_pix(wcsData)
		dat = cutout(hduData[0].data, pixX, pixY, pscale = galDict[galID].pscale)
		whtDat = cutout(hduWht[0].data, pixX, pixY, pscale = galDict[galID].pscale)
		segDat = cutout(hduSeg[0].data, pixX, pixY, pscale = galDict[galID].pscale)
		mask = (segDat!=galDict[galID].id) & (segDat!=0) # mask out other galaxies

		# if everything is nan, go ahead and skip (sometimes true for F140W)
		if np.amin(np.isnan(dat)):
			print('array is NaN for galaxy '+str(galID)+' in filter '+filterName)
			continue
		if np.nanmax(dat) == 0:
			print('array is 0 for galaxy '+str(galID)+' in filter '+filterName)
			continue	

		# replace NaNs in files so galfit doesn't get mad
		dat[np.isnan(dat)] = 0
		whtDat[np.isnan(dat)] = 0

		# write fits for data. multiply dat & sig by wht to get in units of counts
		# fits.writeto('galfitInputs/'+filterName+'_'+str(galID)+'_dat.fits',
		# 	data=dat*exptime,
		# 	header = header, overwrite=True)
		# fits.writeto('galfitInputs/'+filterName+'_'+str(galID)+'_wht.fits',
		# 	data=np.sqrt(np.abs(dat*exptime) + np.divide(exptime**2., whtDat,
		# 	out=np.zeros_like(whtDat), where=np.sqrt(whtDat)!=0)),
		# 	header = header, overwrite=True)
		# fits.writeto('galfitInputs/'+filterName+'_'+str(galID)+'_mask.fits',
		# 	data=mask.astype(int),
		# 	header = header, overwrite=True)
		fits.writeto('galfitInputs/'+group+'_dat.fits',
			data=dat*exptime,
			header = header, overwrite=True)
		fits.writeto('galfitInputs/'+group+'_wht.fits',
			data=np.sqrt(np.abs(dat*exptime) + np.divide(exptime**2., whtDat,
			out=np.zeros_like(whtDat), where=np.sqrt(whtDat)!=0)),
			header = header, overwrite=True)
		fits.writeto('galfitInputs/'+group+'_mask.fits',
			data=mask.astype(int),
			header = header, overwrite=True)
			
		# # determine background level
		# fs = np.zeros((galDict[galID].pscale))
		# apIn = photutils.EllipticalAperture(positions=(galDict[galID].pscale+pixX%1, galDict[galID].pscale+pixY%1),
		# 	a=1, b=galDict[galID].q, theta=galDict[galID].pa)
		# for i, r in enumerate(range(2,galDict[galID].pscale+2)):
		# 	apOut = photutils.EllipticalAperture(positions=(galDict[galID].pscale+pixX%1, galDict[galID].pscale+pixY%1),
		# 	a=r, b=galDict[galID].q*r, theta = galDict[galID].pa)
		# 	maskArea = photutils.aperture_photometry(mask, apOut)['aperture_sum'][0] - \
		# 		photutils.aperture_photometry(mask, apIn)['aperture_sum'][0]
		# 	area = apOut.area() - apIn.area() - maskArea
		# 	fs[i] = (photutils.aperture_photometry(dat*exptime, apOut, mask=mask)['aperture_sum'][0] - \
		# 		photutils.aperture_photometry(dat*exptime, apIn, mask=mask)['aperture_sum'][0]) / area
		# 	apIn = apOut
		# # set sky to mean of pixels past 2nd turnaround of derivative
		# try:
		# 	sky = fs[np.where((fs[1:] - fs[:-1])>0)[0][1]:].mean()
		# except IndexError:
		# 	sky = 0
			
			
		mag = 25 - 5/2. * np.log10(cat[1].data[galDict[galID].id - 1]['f_'+filterName])
		pos = [galDict[galID].pscale+pixY%1, galDict[galID].pscale+pixX%1]
		if np.isnan(mag):
			print('magnitude is NaN for galaxy '+str(galID)+' in filter '+filterName)
			continue

		# WRITE PARAM FILE
		# galfit_write_param('galfitInputs/'+filterName+'_'+str(galID)+'.input', filterName+'_'+str(galID)+'_dat.fits', \
		# 	filterName+'_'+str(galID)+'_output.fits', filterName+'_'+str(galID)+'_wht.fits', "psf_"+survey+".fits", \
		# 	filterName+'_'+str(galID)+'_mask.fits', filterName+'_'+str(galID)+'.constraints', dat.shape, zpt, \
		# 	str(galDict[galID].pscale+pixX%1)[:7], str(galDict[galID].pscale+pixY%1)[:7], str(mag)[:7], \
		# 	str(galDict[galID].re/pixScale)[:7], str(galDict[galID].n)[:7], str(galDict[galID].q)[:7], \
		# 	str(galDict[galID].pa)[:7], sky='0.0')#sky='{:.2f}'.format(sky))
		galfit_write_param('galfitInputs/'+group+'.input', group+'_dat.fits', \
			group+'_output.fits', group+'_wht.fits', "psf_"+survey+".fits", \
			group+'_mask.fits', group+'.constraints', dat.shape, zpt, \
			str(galDict[galID].pscale+pixX%1)[:7], str(galDict[galID].pscale+pixY%1)[:7], str(mag)[:7], \
			str(galDict[galID].re/pixScale)[:7], str(galDict[galID].n)[:7], str(galDict[galID].q)[:7], \
			str(galDict[galID].pa)[:7], sky='0.0')#sky='{:.2f}'.format(sky))		

		# WRITE CONSTRAINTS FILE
		# galfit_write_constraints('galfitInputs/'+filterName+'_'+str(galID)+'.constraints', mag, \
		# 	nmin='0.2', nmax='8.0', remin='0.3', remax='400')
		galfit_write_constraints('galfitInputs/'+group+'.constraints', mag, \
			nmin='0.2', nmax='8.0', remin='0.3', remax='400')
			
		#### actually run galfit
		os.chdir("galfitInputs")
		# return_code = subprocess.call('/Users/wren/galfit/galfit '+
		# 	filterName+'_'+str(galID)+'.input > '+filterName+'_'+str(galID)+'.log', shell=True)
		return_code = subprocess.call('/Users/wren/galfit/galfit '+
			group+'.input > '+group+'.log', shell=True)
		os.chdir("..")

		### read in galfit results
		# if not os.path.isfile('galfitInputs/'+filterName+'_'+str(galID)+'_output.fits'):
		# 	print('No output file found for galaxy '+str(galID)+' in filter '+filterName)
		# 	continue
		# hduOut = fits.open('galfitInputs/'+filterName+'_'+str(galID)+'_output.fits')
		if not os.path.isfile('galfitInputs/'+group+'_output.fits'):
			print('No output file found for galaxy '+str(galID)+' in filter '+filterName)
			continue
		hduOut = fits.open('galfitInputs/'+group+'_output.fits')
		
		# check output flags-- if it crashed, move on
		if ('2' in hduOut[2].header['flags'].split(' ')) or \
			('1' in hduOut[2].header['flags'].split(' ')):
			print('Bad galfit flag for galaxy '+str(galID)+' in filter '+filterName)
			continue
			
		# if it seg faulted, it substitutes a default image which has a diff size-- move on
		# (this happens to like, one galaxy in the whole sample...)
		if hduOut[3].shape != dat.shape:
			print('Seg fault for galaxy '+str(galID)+' in filter '+filterName)
			continue	

		if len(hduOut[2].header['1_mag'].split()) == 3:
			mag = float(hduOut[2].header['1_mag'].split()[0])
			magErr = float(hduOut[2].header['1_mag'].split()[2])
		else:
			mag = float(hduOut[2].header['1_mag'].split()[0][1:-1])
			mag = np.nan
		try:
			re = float(hduOut[2].header['1_re'].split()[0])
			reErr = float(hduOut[2].header['1_re'].split()[2])
		except ValueError:
			re = float(hduOut[2].header['1_re'].split()[0][1:-1])
			reErr = np.nan
		try:
			n = float(hduOut[2].header['1_n'].split()[0])
			nErr = float(hduOut[2].header['1_n'].split()[2])
		except ValueError:
			n = float(hduOut[2].header['1_n'].split()[0][1:-1])
			nErr = np.nan
		if len(hduOut[2].header['1_xc'].split()) == 3:
			xc = float(hduOut[2].header['1_xc'].split()[0])
			xcErr = float(hduOut[2].header['1_xc'].split()[2])
		if len(hduOut[2].header['1_yc'].split()) == 3:
			yc = float(hduOut[2].header['1_yc'].split()[0])
			ycErr = float(hduOut[2].header['1_yc'].split()[2])
		try:
			q = float(hduOut[2].header['1_ar'].split()[0])
			qErr = float(hduOut[2].header['1_ar'].split()[2])
		except ValueError:
			q = float(hduOut[2].header['1_ar'].split()[0][1:-1])
			qErr = np.nan
		try:
			pa = float(hduOut[2].header['1_pa'].split()[0])
			paErr = float(hduOut[2].header['1_pa'].split()[2])
		except ValueError:
			pa = float(hduOut[2].header['1_pa'].split()[0][1:-1])
			paErr = np.nan

		# construct intrinsic sersic profile from galfit results, out to 10kpc
		re_kpc = re*pixScale / cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value
		
		min_r = pixScale / 1./4 / cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value # set min at 1/4 pixel
		rs_kpc = np.logspace(np.log10(min_r), 2.0, 100) # correct first 10kpc, rest just sersic profile
		#rs_kpc = 1.1**np.array(range(1, 1+int(np.ceil(np.log(100/min_r) / np.log(1.1))))) * min_r
		
		rs_pix = rs_kpc / pixScale * cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value
		# this is the flux within each annulus (r 0-1, 1-2, 2-3, ...)
		Ls = sersicLum(mag, rs_pix, re, n, translate[filterName][0])[1:] - \
			sersicLum(mag, rs_pix, re, n, translate[filterName][0])[:-1]
		areas = np.zeros(len(rs_pix))
		apIn = photutils.EllipticalAperture(positions=(xc-1, yc-1), a=rs_pix[0],
			b=rs_pix[0]*q, theta=pa+45)

		aps = []
		flux = []
		error = []
		# measure residual
		for i, r in enumerate(rs_pix[1:]):
			aps.append(apIn)
			apOut = photutils.EllipticalAperture(positions=(xc-1, yc-1),a=r,
				b=r*q, theta=pa+45) # + 90deg is to account for diff definitions of theta in photutils/galfit
			areas[i] = apOut.area() - apIn.area()
			maskArea = (photutils.aperture_photometry(mask, apOut)['aperture_sum'][0] -
				photutils.aperture_photometry(mask, apIn)['aperture_sum'][0])
			corr = areas[i] / (areas[i] - maskArea)
			error.append(emptyApError(filterName, areas[i], whtDat, len(dat)//2, survey) * photflam)

			# don't add residual if more than 10kpc out
			if rs_kpc[i] > 10.0:
				flux.append(flux[-1])
				apIn = apOut
				continue

			# galfit image is in units of e-*1000, so multiply by photflam/1000 to get in Flambda
			flux.append((photutils.aperture_photometry(hduOut[3].data, apOut, mask=mask)['aperture_sum'][0] -
				photutils.aperture_photometry(hduOut[3].data, apIn, mask=mask)['aperture_sum'][0]) \
				* corr * photflam / exptime)

			# next inner aperture is the outer aperture for this annulus
			apIn = apOut

		# for plotting
		rs_pix = rs_pix[1:]
		rs_kpc = rs_kpc[1:]
		areas = areas[:-1]

		# store for later
		sz[galID][filterName] = Ls + np.array(flux) # in Flambda
		sz[galID][filterName+'_err'] = np.array(error) # in Flambda
		sz[galID][filterName+'_area'] = areas # in pix^2
		sz[galID][filterName+'_r'] = rs_pix # in pix
		# sz[galID][filterName+'_ap'] = aps
		
print('Finished galfit runs')

# remove galfit log files
for f in os.listdir('galfitInputs/'):
	if f.startswith('galfit'):
		os.remove('galfitInputs/'+f)

for galID in galDict.keys():
	# make sure there's one filter blue of u and red of g to run EAZY
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

	# now, write output file (ie, .cat file to feed to eazy) for each galaxy
	print('Writing .cat file for galaxy '+str(galID))
	# with open('szomoru_files/'+str(galID)+'.cat', 'w') as f:
	with open('szomoru_files/'+group+'.cat', 'w') as f:
		# write header
		f.write('# id\tz_spec ')
		s = ''
		for im in images:
			s+=translate[im][1] + '\t'
			s+='E'+translate[im][1][1:] + '\t'
		f.write(s+'\n')

		# write each annulus as a separate "galaxy"
		for i in range(len(rs_kpc)):
			f.write(str(i) + '\t' + str(galDict[galID].z) + '\t')
			for im in images:
				if im in sz[galID]:
					f.write('{:.5f}'.format(Fnu_arb_zpt(sz[galID][im][i],
						translate[im][0], zpt=32))+'\t')
					f.write('{:.5f}'.format(Fnu_arb_zpt(sz[galID][im+'_err'][i],
						translate[im][0], zpt=32)) + '\t')
				else:
					f.write('-99\t-99\t')
			f.write('\n')

	# edit zphot.param file to run eazy
	with open('szomoru_files/zphot.param', 'r') as f:
		zphot = f.readlines()
	zphot[27] = 'CATALOG_FILE		  '+group+'.cat		 # Catalog data file\n'
	zphot[33] = 'MAIN_OUTPUT_FILE	  '+group+'	  # Main output file, .zout\n'
	zphot[60] = 'REST_FILTERS		  156\n' # first do u
	with open('szomoru_files/zphot.param', 'w') as f:
		f.writelines(zphot)

	# run EAZY on first rf filter
	os.chdir("szomoru_files")
	# return_code = subprocess.call("./eazy > 'logs/"+str(galID)+"_156.log'", shell=True)
	return_code = subprocess.call("./eazy > 'logs/"+group+"_156.log'", shell=True)
	os.chdir("..")

	# change to other rf filter
	zphot[60] = 'REST_FILTERS		  157\n' # then do g
	with open('szomoru_files/zphot.param', 'w') as f:
		f.writelines(zphot)

	# run EAZY on second rf filter
	os.chdir("szomoru_files")
	# return_code = subprocess.call("./eazy > 'logs/"+str(galID)+"_157.log'", shell=True)
	return_code = subprocess.call("./eazy > 'logs/"+group+"_157.log'", shell=True)
	os.chdir("..")

	# read in EAZY results
	# u = np.nan_to_num(np.loadtxt('szomoru_files/OUTPUT/'+str(galID)+'.156.rf')[:,-1])
	# g = np.nan_to_num(np.loadtxt('szomoru_files/OUTPUT/'+str(galID)+'.157.rf')[:,-1])
	u = np.nan_to_num(np.loadtxt('szomoru_files/OUTPUT/'+group+'.156.rf')[:,5])
	g = np.nan_to_num(np.loadtxt('szomoru_files/OUTPUT/'+group+'.157.rf')[:,5])
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
	lum_g = flux2lum((g * 10**(-32.32)), 3e18/lam_g, galDict[galID].z)
	ug_color = -2.5 * np.log10(u / g)
	ug_color_err = np.sqrt(u_err_mag**2. + g_err_mag**2.)

	# when error in color is >0.2dex, set color value to threshold
	ug_color_corr = ug_color
	ug_color_corr[ug_color_err > 0.2] = ug_color[ug_color_err <= 0.2][-1]

	### find M/L color relation at this redshift
	good = np.where((master[1].data['z_best'] > galDict[galID].z - 0.2) &
		(master[1].data['z_best'] < galDict[galID].z + 0.2) &
		(master[1].data['use_phot'] == 1) & (master[1].data['z_best_s'] < 3))
	if not len(good[0]) > 500:
		good = np.where((master[1].data['z_best'] > galDict[galID].z - 0.5) &
			(master[1].data['z_best'] < galDict[galID].z + 0.5) &
			(master[1].data['use_phot'] == 1) & (master[1].data['z_best_s'] < 3))
		print('Used wider redshift range for galaxy ' +str(galID))
	all_flam_u = Flam((master[1].data['L156'][good] * (10.**(-6.44))), lam_u)
	all_flam_g = Flam((master[1].data['L157'][good] * (10.**(-6.44))), lam_g)
	all_lum_g = flux2lum(all_flam_g, lam_g, master[1].data['z_best'][good])
	all_ug_color = -2.5*np.log10(master[1].data['L156'][good] / master[1].data['L157'][good])
	all_ML = (10**master[1].data['lmass'][good]) / all_lum_g
	slope, intercept = curve_fit(fitfunc, all_ug_color, np.log10(all_ML))[0]
	# error in the M/L determination:
	MLscatter = np.std(np.log10(all_ML) - fitfunc(all_ug_color, slope, intercept))

	# convert EAZY color to M/L then to mass
	ML = 10**fitfunc(ug_color_corr, slope, intercept)
	M = ML * lum_g
	ML_f160 = M / lum_f160
	re = np.interp(np.sum(M)/2., np.cumsum(M), rs_kpc)

	# vary color within error bars to get error on mass profile, re
	#lumg_err = flux2lum(sz[galID]['F160W_err'], galDict[galID].z)
	re_sim = []
	ML_sim = []
	ML_f160_sim = []
	M_sim = []
	for sim in range(1000):
		sim_color = np.random.normal(loc=ug_color_corr, scale = ug_color_err)
		sim_color[ug_color_err > 0.2] = sim_color[ug_color_err <= 0.2][-1]
		#sim_lum = np.random.normal(loc = lum_g, scale = lumg_err)
		sim_ML = 10**(fitfunc(sim_color, slope, intercept) + \
			np.random.normal(0, MLscatter, size=sim_color.shape))
		sim_M = sim_ML * np.random.normal(lum_g, lum_g_err, size=sim_color.shape)
		ML_f160_sim.append(sim_M/np.random.normal(lum_f160, lum_f160_err,size=sim_color.shape))
		re_sim.append(np.interp(np.sum(sim_M)/2., np.cumsum(sim_M), rs_kpc))
		ML_sim.append(sim_ML)
		M_sim.append(sim_M)
	re_err = np.array((np.nanpercentile(re_sim, 16), np.nanpercentile(re_sim,84))) # np.std(re_sim)
	ML_err = np.std(ML_sim, axis=0)
	ML_f160_err = np.std(ML_f160_sim, axis=0)
	M_err = np.std(M_sim, axis=0)
		
	# store for later
	sz_results[galID]['M'] = M
	sz_results[galID]['M_err'] = M_err
	sz_results[galID]['ML'] = ML
	sz_results[galID]['ML_err'] = ML_err
	sz_results[galID]['ML_f160'] = ML_f160
	sz_results[galID]['ML_f160_err'] = ML_f160_err
	sz_results[galID]['Lg'] = lum_g
	sz_results[galID]['re'] = re
	sz_results[galID]['re_err'] = re_err
	flam_u = u * 3e18/(lam_u**2.) * 10**(-32.24)
	flam_g = g * 3e18/(lam_g**2.) * 10**(-32.24)
	sz_results[galID]['uhalf'] = np.interp(np.sum(flam_u)/2., np.cumsum(flam_u), rs_kpc)
	sz_results[galID]['ghalf'] = np.interp(np.sum(flam_g)/2., np.cumsum(flam_g), rs_kpc)
	try:
		sz_results[galID]['rs_pix'] = sz[galID]['F160W_r']
	except KeyError:
		try:
			sz_results[galID]['rs_pix'] = sz[galID]['F140W_r']
		except KeyError:
			sz_results[galID]['rs_pix'] = sz[galID]['F125W_r']	  
	sz_results[galID]['rs_kpc'] = rs_kpc
	# also store for later in galDict
	galDict[galID].re_szomoru = np.array((sz_results[galID]['re'], *sz_results[galID]['re_err'])) # in kpc!!!
	galDict[galID].M_szomoru = np.array((np.sum(sz_results[galID]['M']), np.sqrt(np.sum(sz_results[galID]['M_err']**2.)))) # Mtot and error

	#### PLOTS ####
	# plot surface brightness profile
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

	# plot u-g and mass
	plt.subplot(412)
	plt.errorbar(rs_kpc, ug_color, yerr=ug_color_err, color='black')
	plt.semilogx(rs_kpc, ug_color_corr, color='red')
	plt.ylabel('u-g')
	plt.ylim((0,2.5))
	plt.gca().tick_params(labelbottom='off')
	
	# plot M/Lg
	plt.subplot(413)
	plt.errorbar(rs_kpc, ML, yerr=ML_err, color='black')
	plt.gca().set_xscale('log')
	plt.gca().set_yscale('log')
	plt.ylabel('M/Lg')
	plt.ylim((1e-1, 1e1))
	plt.gca().tick_params(labelbottom='off')
	
	# plot mass
	plt.subplot(414)
	#plt.semilogx(rs_kpc, np.cumsum(M), color='black')
	plt.errorbar(rs_kpc, np.cumsum(M), yerr = np.sqrt(np.cumsum(M_err**2.)), color='black')
	plt.gca().set_xscale('log')
	plt.ylabel('M(<R)')
	plt.xlabel('r (kpc)')
	plt.axvline(re, label='re = '+str(re)[:5]+'+/-'+str(np.mean(re_err)-re)[:5], color='red')
	plt.axhline(10**(fast[1].data['lmass'][np.where(fast[1].data['id'] == galID)[0][0]]), color='grey', alpha=.6, ls='dashed')
	plt.legend(loc='best')
	pdf.savefig()
	plt.close()

pdf.close()
np.savez('szomoru_files/'+savename+'.npz', sz_results=sz_results)
np.savez('savefiles/'+savedict+'.npz', galDict=galDict) # has .re_szomoru [0]=re, [1]=re_err in kpc
