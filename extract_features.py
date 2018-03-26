#Purpose : For CPSC 8100 Intro to AI Project
#Date : 03/16/2018
#By : Mihir Phatak & Vrunal Mhatre



import tensorflow as tf
from scipy import misc
from os import listdir
from os.path import isfile, join
import data_loader
import argparse
import numpy as np
import pickle
import h5py

def main():
# Set arguments to get file from directories
	parser = argparse.ArgumentParser()
	parser.add_argument('/home/vmhatre/vqa_supervised/Data/train/val')
	parser.add_argument('--model_path', type=str, default='/home/vmhatre/vqa_supervised/Data/vgg16.tfmodel')
	parser.add_argument('--data_dir', type=str, default='/home/vmhatre/vqa_supervised/Data')
	parser.add_argument('--batch_size', type=int, default=10)
	


	args = parser.parse_args()
	
	vgg_file = open(args.model_path)
	vgg16raw = vgg_file.read()
	vgg_file.close()
#using a connected graph with data type GraphDef from tensorflow to find connected componenets #within features if any

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(vgg16raw)
#Creating image input variable with 512*512 features as tf placeholder object
	images = tf.placeholder("float", [None, 512, 512, 3])
	tf.import_graph_def(graph_def, input_map={ "images": images })

#Get default connected components
	graph = tf.get_default_graph()
# Defining name-value pair to retreive graphs by operation name
	for opn in graph.get_operations():
		print "Name", opn.name, opn.values()
# Get data from training and validation files
	all_data = data_loader.load_questions_answers(args)
	if args.split == "train":
		qa_data = all_data['training']
	else:
		qa_data = all_data['validation']
#Evaluating total images before building/testing model	
	image_ids = {}
	for qa in qa_data:
		image_ids[qa['image_id']] = 1

	image_id_list = [img_id for img_id in image_ids]
	#print "Total Images", len(image_id_list)
	
	
	sess = tf.Session()
	fc7 = np.ndarray( (len(image_id_list), 4096 ) )
	idx = 0
#For every pixel in size 512*512 storing features in image_batch
	while idx < len(image_id_list):
		image_batch = np.ndarray( (args.batch_size, 512, 512, 3 ) )
#for every image in dataset load image in imagebatch file using load_image_array
		count = 0
		for i in range(0, args.batch_size):
			if idx >= len(image_id_list):
				break
			image_file = join(args.data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(args.split, args.split, image_id_list[idx]) )
			image_batch[i,:,:,:] = utils.load_image_array(image_file)
			idx += 1
			count += 1
		
		
		feed_dict  = { images : image_batch[0:count,:,:,:] }
#Define a Rectified Linear Unit (ReLU)  to store Graph of images which we then mulitply by feed images
		fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
		fc7_batch = sess.run(fc7_tensor, feed_dict = feed_dict)
		fc7[(idx - count):idx, :] = fc7_batch[0:count,:]

		
#Saving fc7 features after extracting from ReLU and image 
	h5f_fc7 = h5py.File( join(args.data_dir, args.split + '_fc7.h5'), 'w')
	h5f_fc7.create_dataset('fc7_features', data=fc7)
	h5f_fc7.close()

	print "Saving image id list"
	h5f_image_id_list = h5py.File( join(args.data_dir, args.split + '_image_id_list.h5'), 'w')
	h5f_image_id_list.create_dataset('image_id_list', data=image_id_list)
	h5f_image_id_list.close()

if __name__ == '__main__':
	main()
