#Purpose : For CPSC 8100 Intro to AI Project
#Date : 03/16/2018
#By : Mihir Phatak & Vrunal Mhatre````


import numpy as np
from scipy import misc
import tensorflow as tf

# VGG 16 accepts RGB channel 0 to 1 (This tensorflow model).
def load_image_array(input):
	img = misc.imread(input)
	# GRAYSCALE
	if len(img.shape) == 2:
		img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
		img_new[:,:,0] = img
		img_new[:,:,1] = img
		img_new[:,:,2] = img
		img = img_new

	img_resized = misc.imresize(img, (224, 224))
	return (img_resized/255.0).astype('float32')
