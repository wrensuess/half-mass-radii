'''This file contains helper functions that will write both a .param and a .constraints file
in the style that GALFIT wants. Essentially, this lets us easily/algorithmically write out
code to run GALFIT. Used for both methods 2 & 3. '''


def galfit_write_param(filename, input, output, sigma, psf, mask, constraints, region, zpt, x, y, mag, re, n, q, pa, sky='0.0'):		
	'''Writes a GALFIT-style parameter file. As inputs, we want to take in values of anything
    that's an option to vary in the .param file. input, output, sigma, psf, mask, and constraints 
    are all strings describing filenames. region is a 2x2 array describing [[xmin, xmax], [ymin, ymax]]
    to fit. zpt is a float with the zeropoint of the image in that filter. x, y, mag, re, n, q, and pa
    are all strings with intial guesses at the galfit fit parameters. '''
    
    # open the file
    f = open(filename, 'w')

    ########################
    # Write the header information
    f.write("===============================================================================\n")
    f.write("# IMAGE and GALFIT CONTROL PARAMETERS\n")
    f.write("A) "+input+'	# Input data image (FITS file)\n')
    f.write("B) "+output+'  # Output data image block\n')
    f.write("C) "+sigma+'	# Sigma image name (made from data if blank or "none")\n')
    f.write("D) "+psf+" # Input PSF image and (optional) diffusion kernel\n")
    f.write("E) 1					# PSF fine sampling factor relative to data\n")
    f.write("F) "+mask+' # Bad pixel mask (FITS image or ASCII coord list)\n')

    f.write("G) "+constraints+' # File with parameter constraints (ASCII file)\n')
    f.write("H) 1 "+str(region[0])+" 1 "+str(region[1]) + " # Image region to fit (xmin xmax ymin ymax)\n")
    f.write("I) 300	   300			# Size of the convolution box (x y)\n")


    f.write("J) "+str(zpt)+"			   # Magnitude photometric zeropoint\n")
    f.write("K) 0.06 0.06  # Plate scale (dx dy)	[arcsec per pixel]\n")
    f.write("O) regular				# Display type (regular, curses, both)\n")
    f.write("P) 0					# Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps\n")
    f.write("\n")

    ########################
    # Write the setup parameters:

    f.write("# INITIAL FITTING PARAMETERS\n")
    f.write("#\n")
    f.write("#	 For object type, the allowed functions are: \n")
    f.write("#		 nuker, sersic, expdisk, devauc, king, psf, gaussian, moffat, \n")
    f.write("#		 ferrer, powsersic, sky, and isophote. \n")
    f.write("#\n")
    f.write("#	 Hidden parameters will only appear when they're specified:\n")
    f.write("#		 C0 (diskyness/boxyness), \n")
    f.write("#		 Fn (n=integer, Azimuthal Fourier Modes),\n")
    f.write("#		 R0-R10 (PA rotation, for creating spiral structures).\n")
    f.write("#\n")
    f.write("# -----------------------------------------------------------------------------\n")
    f.write("#	 par)	 par value(s)	 fit toggle(s)	  # parameter description \n")
    f.write("# -----------------------------------------------------------------------------\n")
    f.write("\n")

    ########################
    # Set up object 1: this is the galaxy. We assume it's a single sersic profile,
    # and give initial guesses as to the location, magnitude, re, n, q, and pa
    
    f.write("# Object number: 1\n")
    f.write(" 0) sersic					#  object type\n")
    f.write(" 1) "+x+' '+y+' 1 1  #	 position x, y\n')
    f.write(" 3) "+ mag + '		1	 #	Integrated magnitude \n')
    f.write(' 4) '+re+"	1  #  R_e (half-light radius)	[pix]\n")
    f.write(' 5) '+n+"	1			 #	Sersic index n (de Vaucouleurs n=4) \n")
    f.write(" 6) 0.0000		 0			#	  ----- \n")
    f.write(" 7) 0.0000		 0			#	  ----- \n")
    f.write(" 8) 0.0000		 0			#	  ----- \n")
    f.write(" 9) "+q+"		1  #  axis ratio (b/a)	\n")
    f.write(" 10) "+pa+"	1	 #	position angle (PA) [deg: Up=0, Left=90]\n")
    f.write(" Z) 0						#  output option (0 = resid., 1 = Don't subtract)\n")

    ########################
    # Set up object 1: this is the sky. We fix it to 0, since the images are all sky-subtracted.
    
    f.write("# Object number: 2\n")
    f.write(" 0) sky					#  object type\n")
    f.write(" 1) "+sky+"	0  #  sky background at center of fitting region [ADUs]\n")
    f.write(" 2) 0.0000		 0			#  dsky/dx (sky gradient in x)\n")
    f.write(" 3) 0.0000		 0			#  dsky/dy (sky gradient in y)\n")
    f.write(" Z) 0						#  output option (0 = resid., 1 = Don't subtract) \n")
    f.write("\n")
    f.write("================================================================================\n")
    f.write("\n")
    f.close()
    
def galfit_write_constraints(constraints, mag, nmin='0.2', nmax='8.0', remin='0.3', remax='400'):
    '''Writes a constraint file for GALFIT to read in. Constraints is a string giving
    the filename we want to save this as (should match constraints keyword in function
    above...). Mag is the integrated magnitude of the galaxy in this filter; we'll constrain
    it to be within 3 magnitudes of the catalog value. We constrain the sersic index
    to be between 0.2 and 8 (following van der Wel), and constrain the half-light radius
    to be within 0.3 and 400 (again, following van der Wel). Both of these are implemented
    as "soft constraints" as is standard for GALFIT. '''
    
	f = open(constraints, 'w')
	f.write('# Component/	 parameter	 constraint	 Comment\n')
	f.write('# operation	 (see below)   range\n')
	f.write('1		n	 '+nmin+' to '+nmax+'	   # Soft constraint\n')
	f.write('\n')
	f.write('1		re	  '+remin+' to '+remax+'	  # Soft constraint\n')
	f.write('\n')
	f.write('1		mag	   '+str(mag-3)[:7]+' to '+str(mag+3)[:7]+ \
		'	 # Soft constraint\n')
	f.close()    