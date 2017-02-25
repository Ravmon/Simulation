#This code equalizes the histogram of all the images in the various samples. Basically what this does is that it spreads out the illumination evenly.

from __future__ import division
import pims
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filter import rank
from scipy.ndimage import *
import pims







a=pims.ImageSequence( '/home/raphael/Documents/python/Colliods/Correlations/Sckit_Image/RUN7_3/Raw_Image/*.png', as_grey=False) # folder for RUN7_3




new_a = [] # this is needed because a=pims.ImageSequence is an object that cannot be replaced.

for index, val in enumerate(a):
  
  #new_a.append(exposure.equalize_adapthist(val))
  #p2, p98 = np.percentile(val, (2, 98))
  #new_a.append(exposure.rescale_intensity(val, in_range=(p2, p98)))
  new_a.append(exposure.equalize_hist(val))
  

plt.imshow(new_a[0])

#plt.show()
for index, val in enumerate(new_a):
    plt.imsave('/home/raphael/Documents/python/Colliods/Correlations/Sckit_Image/RUN7_3/Processed_Image/Processed_Image_%i.png' % index, val) #Folder for RUN7_3
    #plt.imsave('/home/raphael/Documents/python/Colliods/Correlations/Sckit_Image/RUN5_3/Processed_Image/Processed_Image_%i.png' % index, val) #Folder for RUN5_3
    
