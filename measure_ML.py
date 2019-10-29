'''This is where we measure the observed M/L gradient (ie, convolved with the HST F160W PSF).
The general steps are:
1) read in all the relevant files-- the param file telling us what galaxies to look at, 
    all the images, etc, etc
2) measure aperture photometry in all filters where we have resolved HST imaging
3) use FAST to model the SEDs of each elliptical annulus
4) perform an "integral constraint" to make sure that the sum of all the models for all
    the annuli matches the observed photometry in the integrated bands
These steps are described in detail in Suess+19a. 

The output of this code is a dictionary that contains 'gal' objects (class described
in photFuncs.py). At the end of this code, each galaxy has a measured as-observed 
mass profile and M/L profile.

This code also produces a few diagnostic plots that may or may not be useful.

'''


# import libraries
from photFuncs import *
import os
import subprocess
import scipy.io
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from matplotlib.backends.backend_pdf import PdfPages
import time

# set plot settings
plt.interactive(True)
sns.set(style="ticks", font_scale=1.5)
sns.set_style({"xtick.direction":u'in', "ytick.direction":u'in', "lines.linewidth":1.2})

################################ READ INPUTS ################################
''' Start by reading in the input file. This is set up to be run command-line with
the input file provided as an argument, e.g. "python measure_ML.py inputfile_directory/input.param".
Right now the input file format is way too specific for general use-- it matches 
exactly the input files I was generating algorithmically for the paper. Will
be updated soon to instead take a .json file that can be a little less rigorously
formatted...'''

# input file should be given as a command-line argument; if not, exit program
if len(sys.argv) < 2:
	sys.exit('Please list the input parameter file as a command-line argument')
inputfile = sys.argv[1]

# use function in photFuncs.py to parse in the input file and check to make sure
# that all the paths actually exist.
survey, IDs, zs, filterResPath, translatePath, galfitPath, catalogPath, images, \
     filters, rmFASTgrid, pixScale, psf, fastPath, fastDir, fastOut, library, \
     nsim, savedict, imPaths = readInputFile(inputfile)

#### Make translate file  ####
# keys are the names of the filters (as they appear in the 3D-HST survey); 
# values are the central wavelength of the filter and the EAZY/FAST filter number
translate = makeTranslateDict(filterResPath, translatePath)


################################ OPEN FILES ################################
# first get the van der Wel structural parameters from the listed galfit catalog
galfit = np.loadtxt(galfitPath)
		#   3dhst_NUMBER, RA, DEC, f, mag, dmag, re, dre, n, dn, q, dq, pa, dpa, sn
        
# open the 3D-HST photometric catalog (cat) and the 3D-hST FAST catalog (fast)        
cat = fits.open(catalogPath)
fast = fits.open(fastPath)


################################ MEASURE CATALOG FLUXES ################################
'''Here, we want to load in the already-measured values for each galaxy. This includes
the 3D-HST catalog fluxes, the FAST fits to the whole galaxy, and the van der Wel
morphological catalogs. This is also a place to check and make sure that all of the IDs
we provided exist in the catalogs before trying to measure aperture photometry on
non-existing galaxies...'''

# initialize dictionaries to hold the "gal" objects ('galDict') 
# keys will be the ID number, values will be a "gal" instance
galDict = {}

# for each galaxy in our list of galaxies, create a gal object 
# this retrieves the galfit and catalog values
for i, galID in enumerate(IDs):
    galDict[galID] = gal(survey, galID, zs[i], galfit, cat, fast, pixScale, psf, translate)
    
    # make sure that the galaxy actually has measurements in the van der Wel catalog
    # if not, notify the user and 
    if galDict[galID].nAnnuli == 0:
        print('Cannot calculate for galaxy'+str(galID))
        galDict.pop(galID)
        IDs.remove(galID)
        
# print a status update        
print('Got catalog fluxes for '+str(len(galDict.keys()))+' galaxies.')


############################ CALCULATE APERTURE PHOTOMETRY ############################
'''Alright, here's the first big piece of analysis that we're adding to preexisting stuff.
We want to go measure the resolved photometry in elliptical annuli for each galaxy in our
list of targets. To do this, we:
1) calculate aperture photometry in the detection band. We space our annuli 1 PSF HWHM wide
     out until the S/N in the last annulus is < some threshhold (=10 in Suess+19)
2) use those annuli and calculate resolved aperture photometry in all of the other filters
    that we have PSF-matched imaging
3) get errors on all of the resolved photometry. This is done using the empty aperture
    method described in the Skelton+14 3D-HST imaging data release paper.
4) calculate a total aperture correction. This is done in the same way that we do aperture
    corrections for, e.g. fiber spectroscopy: scale the total aperture photometry up by
    a linear factor such that the total measured aperture photometry sums to the same
    value as the total photometry in a large aperture (e.g., 3D-HST catalog values)'''

# make sure we also open up segmentation map (same for all filters): this is output
# from SExtractor that tells us where other galaxies are so we can mask them out
hduSeg = fits.open(imPaths['seg'], ignore_missing_end=True)

# for each filter where we have resolved imaging, measure aperture photometry
for filterName in images:   
    # open images
	hduData = fits.open(imPaths[filterName], ignore_missing_end=True)
	wcsData = WCS(hduData[0].header)
	paData = 0.0 # from F160W; same for all the 3D-HST imaging
	photflam = hduData[0].header['photflam'] # conversion data -> physical units
	hduWht = fits.open(imPaths[filterName+'_wht'], ignore_missing_end=True) # weight map

    # some of the headers (especially in UDS) don't have any photflam values listed
    # we definitely need these, otherwise code reports no flux for those filters.
    # so instead, use the photflam values from COSMOS (this method verified by Barro via email)
	if photflam == 0:
		print('Using photflam from COSMOS for filter '+filterName)
		if filterName == 'F140W': photflam = 1.4737148e-20
		elif filterName == 'F160W': photflam = 1.9275602e-20
		elif filterName == 'F125W': photflam = 2.24834169999999e-20
		elif filterName == 'F606W': photflam = 7.8624958e-20
		elif filterName == 'F814W': photflam = 7.0331885e-20
		# if we don't have a photflam, this is bad enough to force an exit
        else: sys.exit('No photflam found.')

    # calculate photometry
    # first, do the detection band: this makes the list of apertures that
    # we'll use for all the rest of the filters
	if filterName == detectionBand:
		badGals = [] # keep track of galaxies where we can't calculate imaging
		for galID in IDs:
			val = galDict[galID].calcPhotometry_detectionband(filterName, photflam,
				hduData[0].data, wcsData, hduWht[0].data, hduSeg[0].data, paData,
				SNthresh=10.)
                
            # the above function returns -99 if we couldn't do the calculation
            # for some reason (e.g., galaxy is riiiiight at the edge; weight map
            # is infinite, otherwise bad data...). If this is the case, make sure
            # we remove this galaxy from our analysis and make a note of it.    
			if val == -99:
				print('Cannot calculate for galaxy '+str(galID))
				badGals.append(galID)
		# remove all of the 'bad' galaxies
        for badID in badGals:		
			galDict.pop(badID)	
			IDs.remove(badID)
			print('Removed galaxy '+str(badID))
	
    # if we're not in the detection band, use the normal calcPhotometry function
    # to get the resolved aperture photometry
    else:
		for galID in IDs:
			galDict[galID].calcPhotometry(filterName, photflam, hduData[0].data, \
	            wcsData, hduWht[0].data, hduSeg[0].data, paData)

    # close images
	hduData.close()
	hduWht.close()
    
    # update user with status
	print('Measured aperture photometry in filter '+filterName)
# close segmentation map that we opened at the very beginning
hduSeg.close()

#### Calculate correction factor between images and catalog  ####
for galID in IDs:
    galDict[galID].calc_corr(images)
print('Calculated correction factors') # update user


############################ GET MASS IN EACH ANNULUS ############################
'''Our next step is to running the stellar population synthesis (SPS) code FAST
on the photometry we measure in *each annulus*. This differs from the FAST
catalog we read in above-- that's for the integrated light from the whole galaxy.
Here, we just want the derived SPS parameters (stellar age "age", dust extinction
value "Av", and star formation timescale "tau") for the light in the small 
elliptical annuli we created above. FAST is written in IDL, so to keep things
~simple we're going to use python to write out a FAST input file, run FAST
via the command line, then read back in the output files. Then, we're going to
do the integral constraint (inspired by Wuyts+12) that adjusts the best fits
for each annulus until the sum of the modeled annuli fluxes matches the integrated
photometry measured for the rest of the galaxy. This is how we fold in the fact
that each galaxy has measurements in many other filters than just the 5-8 bands
that have resolved HST imaging.
This is the last step! once we've done this, we have an as-observed M/L profile
for the galaxy. Then, we'll move to other files to do a few different modeling
techniques to interpret the results. '''

# write FAST catalog
write_FAST_catalog('FAST/'+fastOut, filters, translate, images, galDict, IDs)
print('Wrote FAST catalog to '+'FAST/'+fastOut) # update the user

# give the computer 2sec to make sure the file has actually finished writing
# before we try and call it (I sometimes got errors otherwise...)
time.sleep(2) 

# make a 'param' file that tells FAST what to do. We only need to change the 
# 'catalog' line in the FAST param file to the specified catalog
with open('FAST/fast.param') as f:
	lines = f.readlines()
lines[80] = "CATALOG        = '" +fastOut[:-4]+ "'\n"
# and write it back out to the FAST directory
with open('FAST/fast.param', 'w') as f:
	f.writelines(lines)

# actually run fast-- change to the right directory, and use python
# to call fast command-line
os.chdir("FAST")
print("Starting FAST")
return_code = subprocess.call(fastDir+"fast fast.param", shell=True)
print("Finished running FAST")
os.chdir("..")

#### Read in FAST results ####
'''We get two different things out from my (slightly hacked....) version of 
FAST. First, we get the 'grid' that has the unscaled expected photometry
for all possible (age,tau,Av) combos. This is what FAST builds before it
does the reduced chi^2 optimization. We want this because we're going to
let the best-fit annulus (age,tau,Av) float a little to do the integral
constraint suggested by Wuyts+12 and detailed in Suess+19-- in essense,
this constraint make sure that the sum of the fits to all the annuli matches
the integrated photometry in every measured band. FAST gives us one grid
for the entire set of galaxies we fit.
The second thing that FAST gives us is a bunch of things that are specific
to each "galaxy" (where in our case, each "galaxy" is actually one annulus
in a galaxy). We'll also need those... '''

# find the FAST folder: it should be the most recently-created folder in the FAST subdir
folders = list(filter(lambda x: os.path.isdir(x), ['FAST/'+i for i in os.listdir('FAST/')]))
folders.sort(key=lambda x: os.path.getmtime(x))
folders = [i for i in folders if i.startswith('FAST/'+library)]
folder = np.array(folders)[[survey in fname for fname in [os.listdir(f)[0] for f in folders]]][0]

# read in the .fout file (has the best-fit parameters for each annulus)
fout = np.loadtxt('FAST/'+fastOut[:-4]+'.fout')	

# read in the grid file (this is built from the libraries FAST calls)
grid = scipy.io.readsav(folder+'/grid.save')
grid['grid'] = grid['grid'][:,:,:,0,:,:]		# remove redundant metallicity

# read in the FAST results for each annulus
for galID in IDs:
    galDict[galID].read_fast(IDs, folder, fastOut, grid)
    print('Read in FAST results for galaxy '+survey+str(galID))
    
    # and then get the mass (and M/L) in all the annuli for that galaxy
    # this function also does Monte Carlo error bars to get uncertainties
    # on the measured mass and M/L
    galDict[galID].get_masses(filters, nsim, grid, images, translate)
    print('Found masses and errors for galaxy '+survey+str(galID))

# if we wanted to, remove FAST grid & files it contains
if rmFASTgrid == 'True':
    for i in os.listdir(folder):
        os.remove(os.path.join(folder, i))
    os.rmdir(folder)    


############################ DIAGNOSTIC PLOTS ############################
'''For any complicated code, it's important to make some diagnostic plots to 
make sure that everything makes sense. These are the ones that I found helpful;
they are obviously not essential. The only thing left in this piece of code that
isn't plotting is saving the results (important!) so if you want to run this
but comment out the plots make sure to leave in lines 425-438... '''


######## mass profiles plot ########
# diagnostic plot of both the mass profiles (top panel) and M/L gradients (bottom
# panel). For the mass profile, plot both the actual mass profile and the scaled
# H-band light profile.

# open a pdf-- we'll just plot each galaxy as a different page of the pdf
with PdfPages('Plots/'+savedict+'.pdf') as pdf:
	for galID in galDict.keys():
        # first subplot:
		plt.figure()
		plt.subplot(211)
		
        # plot the cumulative measured mass as a function of radius
		plt.errorbar(galDict[galID].edges*pixScale/cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value, 
            np.cumsum(galDict[galID].bestMasses),
			np.sqrt(np.cumsum(galDict[galID].massErrors**2.,axis=1)),
			label='multi-band: re = '+str(re_allbands)[:5], color='black')
        # find the radius where the mass reaches half its max; plot it as a vertical line    
		re_allbands = np.interp(np.sum(galDict[galID].bestMasses)/2.,
			np.cumsum(galDict[galID].bestMasses), 
            galDict[galID].edges*pixScale/cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value)    
		plt.axvline(re_allbands, color='black')
        
        # plot a horizontal line at the mass value that was inferred for the integrated photometry
        # for the whole galaxy. (e.g., check that the sum of the annuli masses matches the total mass!)
		plt.axhline(10**(fast[1].data['lmass'][np.where(fast[1].data['id'] == galID)[0][0]]), 
            color='grey', ls='dashed', alpha=.6)
            
        # plot the H-band light profile, scaled to match the total mass. This gives us a sense 
        # of how important color gradients are for this galaxy.    
		plt.errorbar(galDict[galID].edges*pixScale/cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value, 
            np.cumsum(galDict[galID].photometry['F160W'])* 
			np.sum(galDict[galID].bestMasses)/np.sum(galDict[galID].photometry['F160W']),
			yerr=np.sqrt(np.cumsum(np.array(galDict[galID].photometry['F160W_err'])**2.))*
			np.sum(galDict[galID].bestMasses)/np.sum(galDict[galID].photometry['F160W']),
			label='H only: re = '+str(re_H)[:5], color='Teal')
        # calculate & plot the H-band half-light radius in the same way as half-mass radius above to compare
        # (this isn't the real half-light radius, just an eazy approx given what we've got)   
		re_H = np.interp(np.sum(galDict[galID].bestMasses)/2.,
			np.cumsum(galDict[galID].photometry['F160W'])*np.sum(galDict[galID].bestMasses)/ 
			np.sum(galDict[galID].photometry['F160W']), 
            galDict[galID].edges*pixScale/cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value)             
		plt.axvline(re_H, color='Teal')

        # also plot the vdW catalog galfit half-light radius (accounts for the PSF, so it's
        # on a different scale but can be useful to see...)
		if galDict[galID].galfitflag == 0:
			plt.axvline(galDict[galID].re/cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value,
				color='grey', label='galfit: re = '+
				str(galDict[galID].re/cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value)[:5])
		plt.legend(loc='best')
		plt.title(galID)
		plt.xlabel('r (kpc)')
		plt.ylabel('Mass (Msun)')

        # second plot: plot the M/L directly. Can be used to see if there is a positive/negative/no
        # color gradient. Has to be modeled further to interpret, but useful to see at this stage too.
		plt.subplot(212)
		plt.errorbar(galDict[galID].edges*pixScale/cosmo.arcsec_per_kpc_proper(z=galDict[galID].z).value,
			galDict[galID].bestML, yerr = galDict[galID].MLerr, color='black', fmt='o')
		plt.xlabel('r (kpc)')
		plt.ylabel('M/L')

        # save the figure to the pdf and close so we can go to the next galaxy.
		pdf.savefig()
		plt.close()


######## SED plot ########
# plot both the original and the post-integral-constraint SEDs for each annulus.
# this is useful to make sure that the constraint process is working, and actually
# makes the SED in the WISE bands match the sum of all the annuli SEDs.

# again, we'll just make one big pdf that has a page for each galaxy
with PdfPages('Plots/'+savedict+'_sed.pdf') as pdf:
    for x in galDict.values():
        plt.figure()
        plt.subplot(211)
        
        # first plot: plot the post-integral-constraint SEDs
        plt.title('Constrained '+str(x.id))
        model = np.zeros(x.grid[0].shape[0]) # this will hold the sum of the models for all annuli
        colors_b = sns.color_palette('Blues', x.nAnnuli)
        colors_r = sns.color_palette('Reds', x.nAnnuli)
        
        # plot all the annuli
        for ann in range(x.nAnnuli):
            # model points
            plt.plot(sorted([translate[i][0] for i in filters]), x.grid[ann][(slice(None),) + 
                tuple(x.bestPos[ann])][np.array([translate[i][0] for i in filters]).argsort()],
                color=colors_b[ann])
            model = model + x.grid[ann][(slice(None),)+x.bestPos[ann]]
           
            # observed points
            for fil in images:
                plt.errorbar(translate[fil][0], x.photometry[fil][ann], yerr=x.photometry[fil+'_err'][ann],
                    color=colors_b[ann], marker='*', markersize=10, zorder=11)
       
        # plot the sum of all the models
        plt.plot(sorted([translate[i][0] for i in filters]), 
            model[np.array([translate[i][0] for i in filters]).argsort()], color='black')
            
        # plot the catalog fluxes in *all* filters (not just ones w/ measured photometry)    
        plt.scatter([translate[i][0] for i in filters],
            [x.catalogFlux['f_'+i.lower()] for i in filters], color='red', marker='*')
        
        # make sure the scale makes sense
        plt.gca().set_yscale('log')
        plt.ylim((1e-22, 1e-17))

        # second plot: pre-constraint (straight out of fast) annuli SEDs
        plt.subplot(212)
        plt.title('Original')
        model = np.zeros(x.grid[0].shape[0])
        # again, plot each annulus and add the model to the sum of all models
        for ann in range(x.nAnnuli):
            # model points
            plt.plot(sorted([translate[i][0] for i in filters]), x.grid[ann][(slice(None),) + 
                tuple(x.fastPos[ann])][np.array([translate[i][0] for i in filters]).argsort()],
                color=colors_b[ann])
            model = model + x.grid[ann][(slice(None),)+tuple(x.fastPos[ann])]
            
            # observed points
            for fil in images:
                plt.errorbar(translate[fil][0], x.photometry[fil][ann], yerr=x.photometry[fil+'_err'][ann],
                    color=colors_b[ann], marker='*', markersize=10, zorder=11)
       
        # plot the sum of all the models
        plt.plot(sorted([translate[i][0] for i in filters]), model[np.array([translate[i][0] for i in filters]).argsort()], color='black')
        
        # plot the catalog fluxes in *all* filters (not just ones w/ measured photometry)    
        plt.scatter([translate[i][0] for i in filters],
            [x.catalogFlux['f_'+i.lower()] for i in filters], color='red', marker='*')
        
        # make sure the scale makes sense
        plt.gca().set_yscale('log')
        plt.ylim((1e-22, 1e-17))
        
        # save this galaxy's plot and move to the next one
        pdf.savefig()
        plt.close()


############################ SAVE OUTPUTS ############################
'''We obviously want to save all of our hard work! But first, make sure to
remove all of the unnecessary and HUGE FAST grids that we were lugging
around. If we don't do this, it's really unfeasable to actually run this
code for a large number of galaxies... '''

# remove the grids
for galID in galDict.keys():
    galDict[galID].chiGrid = np.nan
    galDict[galID].massGrid = np.nan
    galDict[galID].grid = np.nan
    
# and save the results!!    
np.savez('savefiles/'+savedict+'.npz', galDict=galDict)
