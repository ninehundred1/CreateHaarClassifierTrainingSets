import cv2
from tkFileDialog import askopenfilename, askdirectory
import os
import sys
from os import listdir
from os.path import isfile, join
import numpy as np
from random import randint

"""
Module that created training sets for Haar classifiers out of sets of images.
With an appropriate background image the module can detect and extract the object
of interest and replace the background with random images to increase the number 
of training images. The object gets also rotated and distorted slightly to further
increase variability of the training images.

The position of the object in each image gets automatically extracted and the bounding
coordinates together with the filenames are parsed into a txt file which can be directly
used to train OpenCV Haar classifiers.

For instruction, press '0' when running this file or see 
https://github.com/ninehundred1/CreateHaarClassifierTrainingSets.git


See bottom of file for further help.

2016 Stephan Meyer fuschro@gmail.com
"""

def update_negatives_filename_list():
	'''
	asks user to specify a folder from which a new negatives.txt file gets created.

	parameter:
	none
	returns:
	writes to file
	'''
	filename = askopenfilename() 
	txt = open(filename)
	newpath = "/".join(filename.split('/')[:-1])
	with open(newpath+'/negatives.txt', 'w') as out_file:
		for item in txt:
			out_file.write("%s\n" % item[0])
	print 'updated negatives.txt in specified folder.'


def convert_images_to_negatives_set():
	'''
	asks user to specify image folder which images gets converted to random negatives
	including negatives.txt file.

	parameter:
	none
	returns:
	writes folder, images and txt file
	'''
	print 'please select folder with images.'
	dir_name_ran_bg, file_names_list_ran_bg = get_filelist_from_user()
	newpath = "/".join(dir_name_ran_bg.split('/')[:-1])+'/negative_images'
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	file_names_save_list = []
	for i, file in enumerate(file_names_list_ran_bg):
		path = dir_name_ran_bg+'/'+file
		print path
		image = cv2.imread(path)
		image = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
		image = cv2.resize(image, (640, 478)) 
		cv2.imwrite(newpath+'/'+"neg%d.jpg" % i, image)	
		file_names_save_list.append('negative_images/neg%d.jpg' %i )
	with open(newpath+'/negatives.txt', 'w') as out_file:
		for item in file_names_save_list:
			out_file.write("%s\n" % item)
	print 'created negative_images folder including negatives.txt file.'


def get_video_source():
	'''
	asks user to load a video file to extract the frames of as image source.

	parameter:
	none
	returns:
	the camera object to read frames from and the frames per second (default to 24)
	'''
	print 'please select source movie'
	filename = askopenfilename() 
	save_name = os.path.basename(filename)
	if not save_name:
		print 'user load Movie path invalid'
		sys.exit()
	print(save_name)
	camera = cv2.VideoCapture(filename)
	fps = camera.get(cv2.cv.CV_CAP_PROP_FPS)
	if fps == 0.0:
		 fps = 24
	return camera, fps


def get_bg_image():
	'''
	asks user to load a background image used for substraction of background.

	parameter:
	none
	returns:
	background Image file
	'''
	print 'please select bg image for substraction (one image that contains BG only)'
	bgname = askopenfilename() 
	if not bgname:
		print 'user load BG image path invalid'
		sys.exit()
	return bgname


def get_filelist_from_user():
	'''
	asks user to load a folder with images.

	parameter:
	none
	returns:
	Image folder and file list
	'''
	dir_name = askdirectory() 
	if not dir_name:
		print 'user load path for images invalid'
		sys.exit()
	file_names_list = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
	print file_names_list[:5]
	print 'total files: %s' %len(file_names_list)
	return dir_name, file_names_list

def create_empty_npstack_from_images(dir_name, file_names_list, delta):
	'''
	reads images with a certain delta spacing and creates an empty Numpy stack of the correct sizes.

	parameter:
	dir_name(string): name of directory of files
	file_names_list(list): all the file file_names_list
	delta(int): spacing of read (if 4, only every 4th image is averaged into mean background)
	returns:
	numpy array of images to be averaged
	'''
	temp_image = cv2.imread(dir_name+'/'+file_names_list[0])
	width, height = temp_image.shape[:2]
	return np.empty((width, height, int(len(file_names_list)/delta)+1))

	
def remove_bg_loop(dir_name, file_names_list, image_stack, use_max, delta):
	'''
	load the first and then the next image with delta offset. Delete loaded imaged from list.
	make new img folder to save the images. Each image is a max or min projection of several
	images, so if you have a static background you should be left with just the background in each image.
	It will also write a negatives.txt file to use for training.

	parameter:
	dir_name(string): name of directory of files
	file_names_list(list): all the file file_names_list
	image_stack(np array): empty stack that matches the data dimensions
	use_max(boolean): max if you want a maximal gray value projection, False, if min projection
	delta(int): spacing of read (if 4, only every 4th image is averaged into mean background)
	returns:
	writes the image with removed bg to file.
	'''
	newpath = dir_name+'/negative_images'
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	processed_batch = 0
	image_counter = 0
	list_entries_to_delete = []
	list_files = []
	while file_names_list:
		for i in range (0, len(file_names_list), delta-processed_batch-1):
			image = cv2.imread(dir_name+'/'+file_names_list[i])
			#convert to 8bit
			image = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
			image_stack[:,:,image_counter] = image 
			image_counter = image_counter + 1
			list_entries_to_delete.append(file_names_list[i])
		#delete list entries
		updated_list = [i for i in file_names_list if i not in list_entries_to_delete]
		if use_max == True:
			mean_img = np.max(image_stack, axis=2)
		else:
			mean_img = np.min(image_stack, axis=2)
		cv2.imwrite(newpath+"/nobg%d.jpg" % processed_batch, mean_img)
		image_counter = 0
		file_names_list = updated_list
		list_entries_to_delete = []
		list_files.append('negative_images/nobg%d.jpg' % processed_batch)
		processed_batch = processed_batch + 1
		print 'processed batch: %s' %processed_batch
	#write file for training
	with open(newpath+'/negatives.txt', 'w') as out_file:
		for item in list_files:
  			out_file.write("%s\n" % item)


def create_empty_background_from_images_for_training(use_max = True, delta = 500):
	'''
	Go through a folder of images with static background, then take several image with a spacing
	of delta and average those either by maximum or by minimum. This will remove all the variance,
	meaning everything that is not consistent between images (in the case of static backgrounds, 
	the foreground)

	parameter:
	use_max(boolean): max if you want a maximal gray value projection, False, if min projection
	delta(int): spacing of read (if 4, only every 4th image is averaged into mean background)
	returns:
	calls other functions and writes the image with removed bg to file.
	'''
	dir_name, file_names_list = get_filelist_from_user()
	stack = create_empty_npstack_from_images(dir_name, file_names_list, delta)
	remove_bg_loop(dir_name, file_names_list, stack, use_max, delta)


def parse_movie_frames_to_file():
	'''
	Load user movie file and save each frame as jpg image in a new folder in the same directory

	parameter:
	none, asks user for video
	returns:
	makes new folder in directory of movie and saves images in that folder
	'''
	camera, fps, width, height, bg_image, dir_name, file_names_list = initialize_image_streams(use_movie)
	newpath = dir_name+'/extracted_images'
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	image_counter = 0

	while True:
		success,image = camera.read()
		if not success:
			break
		image = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )
		cv2.imwrite(newpath+"/extr%d.jpg" % image_counter, image)
		image_counter += 1
	print 'extracted movie into %s images' %image_counter



def initialize_image_streams(use_movie):
	'''
	set up from user the source video file, the bg image for threshold and get sized

	parameter:
	use_movie (boolean): use movie or folder with images
	asks user to supply paths for movie and bg image
	returns:
	camera object
	fps(int): frames per second
	width, height: size of Image
	bg_image: the background image
	'''
	camera = fps = dir_name = file_names_list = None
	if use_movie:
		camera, fps = get_video_source()[0:2]
	else:
		dir_name, file_names_list = get_filelist_from_user()
	bg_image_path = get_bg_image()
	bg_image = cv2.imread(bg_image_path,0)
	cv2.resize(bg_image, (width, height)) 
	#get one image to get size of movie
	success,image = camera.read()
	width = np.size(image, 1)
	height = np.size(image, 0)
	return camera, fps, width, height, bg_image, dir_name, file_names_list


def create_directory():
	'''
	set up directory /positive_images to save images in in current directory.
	If already existing, nothing happens

	parameter:
	none
	returns:
	path to the new directory
	'''
	newpath = "/".join(bg_image_path.split('/')[:-1])+'/positive_images'
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	return newpath


def rotate_image(image):
	'''
	Roate image a random degree between -45 and 45 degrees

	parameter:
	image: the current image
	returns:
	roated image
	'''
	degrees = randint(-45,45)
	center = (width / 2, height / 2)
	M = cv2.getRotationMatrix2D(center, degrees, 1.0)
	try:
		rotated = cv2.warpAffine(image, M, (width, height), borderValue = (0,0,0))
	except Exception:
		print('error in rotate')
		continue
	#cv2.imshow('rotated', rotated)
	#cv2.waitKey(0)
	return rotated


def distort_image(image_rotated):
	'''
	Distorts image by random amount between 0% and 10% of size in all directions.

	parameter:
	image: the current image
	returns:
	distorted image
	'''
	pts_origin = np.float32([[width*(randint(0,1)/float(10)),height*(randint(0,1)/float(10))],
		[width*(randint(9,10)/float(10)),height*(randint(0,1)/float(10))],
		[width*(randint(0,1)/float(10)),height* (randint(9,10)/float(10))],
		[width* (randint(9,10)/float(10)),height* (randint(9,10)/float(10))]])
	pts_dest = np.float32([[0,0],[width,0],[0,height],[width,height]])
	M = cv2.getPerspectiveTransform(pts_origin,pts_dest)
	try:
		distorted = cv2.warpPerspective(rotated,M,(width,height))
	except:
		print('error in distort')
		continue
	#cv2.imshow('distorted', distorted)
	#cv2.waitKey(0)
	return distorted


def substract_bg(image, bg_image):
	'''
	Blurs Images and substracts the bg_image from image

	parameter:
	image: the current image
	bg_image: the image to distract
	returns:
	delta image: the image with the background removed
	image_blurred: blurred image
	image_blurred_copy: copy of blurred

	'''
	#mark object and get bounding coordinates for training Haas
	image_blurred = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY )
	image_blurred_copy = image_blurred
	image_blurred = cv2.medianBlur(image_blurred,3)
	delta_image = bg_image-image_blurred
	#cv2.imshow('substractBG', delta_image)
	#cv2.waitKey(0)
	return delta_image, image_blurred, image_blurred_copy


def threshold_image(delta_image):
	'''
	Binary thresholds Image

	parameter:
	delta_image: the delta image
	returns:
	thresholded image
	'''
	ret,thresh = cv2.threshold(delta_image,70,255,cv2.THRESH_BINARY)
	#cv2.imshow('threshold', thresh)
	#cv2.waitKey(0)
	return thresh


def find_largest_contour(image_mouse, contours):
	'''
	Draws all contours and picks the largest contour from list

	parameter:
	image_mouse: image for contour illustration
	contours: all found contours
	returns:
	contours(poly): the contours with the largest area
	'''
	image_mouse2 = image_mouse
	cv2.drawContours(image_mouse2, contours, -1, (255,255,0), 3)
	#cv2.imshow('contours', image_mouse2)
	#cv2.waitKey(0)
	areas = [cv2.contourArea(c) for c in contours]
	max_index = np.argmax(areas)
	return contours[max_index]


def make_bounding_polygon(cnt):
	'''
	converts polycoordinates to numpy array for drawing

	parameter:
	cnt: contour coordinates
	returns:
	np array of image size
	'''
	poly = []
	for i in cnt:
		poly.append([i[0][0], i[0][1]])		
	return np.array(poly)


def replace_with_rdm_bg(file_names_list_ran_bg, dir_name_ran_bg, width, height, poly,image_mouse_copy):
	'''
	Picks a random image from the file list and uses the boundaries of object to copy
	the object into the random background image

	parameter:
	file_names_list_ran_bg: file list of random images
	dir_name_ran_bg: directory for random image
	width, height: size of image
	poly: bounding polygon of object
	image_mouse_copy: the original image with object
	
	returns:
	image with background replaced
	'''
	#load random bg image
	random_index = randint(0, len(file_names_list_ran_bg)-1)
	path_to_rand = dir_name_ran_bg + '/'+file_names_list_ran_bg[random_index]
	random_bg_image = cv2.imread(path_to_rand,0)
	resized_random_bg_image = cv2.resize(random_bg_image, (width, height)) 
	#get ROI part from original and replace bg
	#fill outside with random image
	mask = resized_random_bg_image.copy()
	mask.fill(255)
	cv2.fillPoly(mask, [poly], 0)
	mask_out=cv2.subtract(mask,resized_random_bg_image)
	mask_out=cv2.subtract(mask,mask_out)
	#fill mouse with mouse
	mask2 = resized_random_bg_image.copy()
	mask2.fill(0)
	cv2.fillPoly(mask2, [poly], 255)
	mask_out2=cv2.subtract(mask2,image_mouse_copy)
	mask_out2=cv2.subtract(mask2,mask_out2)
	image = cv2.add(mask_out, mask_out2)
	#cv2.imshow('replaced', image)
	#cv2.waitKey(0)
	return image


def append_to_txt_file(x,y,w,h, count):
	'''
	Appends image file name and bounding box of object to the 
	positives.txt file needed for training

	parameter:
	x,y,w,h: bounding box coordinates
	count: current image number
	returns:
	modifies the list as list is passed by reference
	'''
	coor_string = '\t'.join([str(x),str(y), str(w),str(h)])
	list_entry = 'positive_images/frame%d.jpg' %count +'\t1\t'+coor_string
	list_coords.append(list_entry)


def write_txt_file(newpath, list_coords):
	'''
	Writes file positives.txt to disk

	parameter:
	newpath: save path
	list_coords: the list
	returns:
	writes to disk
	'''
	with open(newpath+'/positives.txt', 'w') as out_file:
		for item in list_coords:
  			out_file.write("%s\n" % item)


def parse_images(use_movie = False, replace_ran_bg = True):
	'''
	Go through a folder of images or frames in movie, threshold each iamge, finds the biggest contour
	that fits an object of interest, optionally replaces the image around the contoured object with 
	a random background and then draws a bounding box around the object.
	Saves the image on file and creates a text file with the image file name and coordinates of 
	the bounding box.

	parameter:
	use_movie (boolean): use movie or folder with images
	replace_ran_bg(boolean): replace bg with random bg or not
	returns:
	makes new folder in directory of movie and saves images in that folder.

	'''
	
	camera, fps, width, height, bg_image, dir_name, file_names_list = initialize_image_streams(use_movie)
	if replace_ran_bg == True:
		print 'please select folder with random images'
		dir_name_ran_bg, file_names_list_ran_bg = get_filelist_from_user()
	count = 0
	cycle = 1
	num_cycles = 0
	list_coords = []
	newpath = create_directory()
	if not use_movie:
		num_cycles = raw_input('how many times do you want to re-use each image? (try to get 1000 or so total):')

	while True:
		if use_movie:
			success,image = camera.read()
		else:
			image = cv2.imread(dir_name+'/'+file_names_list[count])
			if count == len(file_names_list)-1:
				count = 0
				cycle += 1
			success = [False, True][cycle == num_cycles] 

		if not success:
			break
		image_rotated = rotate_image(image)
		image_distorted = distort_image(image_rotated)
		delta_image, image_mouse, image_mouse_copy = substract_bg(distorted, bg_image)
		thresh = threshold_image(delta_image)
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		cnt = find_largest_contour(image_mouse, contours)
		x,y,w,h = cv2.boundingRect(cnt)
		poly = make_bounding_polygon(cnt)
		cv2.fillPoly(image_mouse, pts =[poly], color=(255,255,255))
		#cv2.imshow('fill largerst', image_mouse)
		#cv2.waitKey(0)
		if replace_ran_bg == True:
			image = replace_with_rdm_bg(file_names_list_ran_bg, dir_name_ran_bg, width, height, poly,image_mouse_copy)
		cv2.imwrite(newpath+'/'+"frame%d.jpg" % count, image)	
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		#cv2.imshow('for training', image)
		#cv2.waitKey(0)
		append_to_txt_file(x,y,w,h, count, coor_string, coor_string)
		count += 1
		print 'converted image: %s' % count 
		#break if ESC   
		if cv2.waitKey(10) == 27:
			break
	write_txt_file(newpath,list_coords)


def show_instructions():
	help_text =  ("WHAT YOU NEED TO GENERATE HAAR CLASSIFIER TRAINING SETS:\n"
	"\n"
	"A. A folder of OBJECT IMAGES that has your target object over the same white/black/etc background, \n"
	"   or a movie where the background remains constant (doesn't move) or has the same color throughout.\n"
	"   These images are for your positives training set.\n"
	"   You probably should have around 1000-2000 of those images (if not using a movie). \n"
	"   If you only have a few (20 or 50) you should run the option where you replace the background with \n"
	"   random images (4) and run it a few times.\n"
	"   Each time you run it the images come out different, so you will end up with different images made\n"
	"   from the same few initial images.\n"
	"\n"
	"B  You also need one BACKGROUND IMAGE that is only the background with ideally the same size as other\n"
	"   images. This image is used to automatically find the borders of your object which is required for \n"
	"   training, so the algorithm knows where your object of interest is.\n"
	"   If you have images of your object over white background, one all-white image is fine. For a movie with a            stationary background or color, you can create clean images of the background using option 5 or 6.\n"
	"  \n"
	"\n"
	"C. A folder with RANDOM IMAGES to form your negatives training set. You can use photos from your phone, \n"
	"   etc. Again, try to have 1000-2000.\n"
	"   If you use the option to relace the background behind your object (mentioned in A), you need two sets\n"
	"   of image, each around 1000-2000 images.\n"
	"\n"
	"\n"
	"\n"
	"\n"
	"To generate the training data sets for the Haar Classifiers from the images:\n"
	"\n"
	"Steps:\n"
	"\n"
	"Make Positives set:\n"
	"a) If you have a folder of a few images (50 or so) with your object over a smooth background (eg white or black etc)\n"
	"use option 4 to create your positive training set. You will also need to supply one image of just the \n"
	"background (same color as your background) and a folder that has many random images.\n"
	"You will then be asked how often to cycle through those object images. Choose it to get you to 1000-2000 \n"
	"images total.\n"
	"\n"
	"b) If you have a folder with a lot images and the same background (such as extracted from a movie file), select\n"
	"option 2. This will not replace the background but only mark the object. For that, you need again supply an \n"
	"image with just the background for extraction.\n"
	"\n"
	"c) If you have a movie on a steady background (some kind of fixed camera), first create a background image out\n"
	"of the movie using option 5 or 6 (which will create a negatives set also, which you might not need),\n"
	"depending if your background is brighter than your object or darker.\n"
	"This will create a few images with just the background. Then you have the option to create a positives training\n"
	"set straight out of the movie by using option 1 or by replacing the background using option 3. \n"
	"Use one of the background images generated before as a background image. If using option 3, then select a\n"
	"folder that contains a lot of random images).\n"
	"\n"
	"\n"
	"Make Negatives set:\n"
	"d) If you have a movie with a steady background you can use option 5 or 6 to create a set of images of just\n"
	"the background. This might not be ideal as there is very little variance between the images, which is not\n"
	"what you really want (you want to train the classifier on many things that are not your object, not few).\n"
	"\n"
	"e) Ideally you just use a lot of random images, as that tells the classifiers many things that are not \n"
	"what to look for. Use option 7 to create a training set out of a folder with many random images.\n"
	"Use 1000-2000 random images in one folder.\n"
	"\n"
	"\n"
	"Other options:\n"
	"You can extract a movie into individual jpg frames by using option 8. You can then use those for b),\n"
	"if you are only interested in certain frames of a movie.\n"
	"\n"
	"If you want to use random images for your negatives set in e), and later want to add more images, you can just\n"
	"create a new txt file that includes all images in the folder by using option 9.\n"
	"\n"
	"\n"
	"\n"
	"USING THE TRAINING SETS\n"
	"\n"
	"1. copy opencv_createsamples.exe and opencv_traincascade from the opencv folder you dowloaded initially\n"
	"   (check folder C:\\Users\\Meyer\\Downloads\\opencv\\build\\x64\\vc10\\bin)\n"
	"\n"
	"   I also copied all dll files from that folder into the same directory as well to the same directory\n"
	"   with the folders positive_images and negative_images. Those two folders also contain the files \n"
	"   positives.txt and negatives.txt, which you need to move one folder up.\n"
	"\n"
	"   So in your working folder you should now have:\n"
	"\t\tFolder negative_images\n"
	"\t\tFolder positive_images\n"
	"\t\tTxt file negatives.txt\n"
	"\t\tTxt file positives.txt\n"
	"\t\topencv_createsamples.exe \n"
	"\t\topencv_traincascade\n"
	"\t\t(and if error the other files from the bin folder)\n"
	"\n"
	"\n"
	"2. Create a vector file of the images in the windows command window.\n"
	"   While in the above directory run (but change num to your image number of positve images):\n"
	"\n"
	"opencv_createsamples -info positives.txt -bg negatives.txt -vec cropped.vec -num 1279 -w 48 -h 48\n"
	"\n"
	"  This creates a vector file called cropped.vec\n"
	"\n"
	"\n"
	"3. Train the cascade using above vector file (can take hours to days). \n"
	"   Again on command while in the same directory run below, but change the number of images to less \n"
	"   what you have (I use 500 as negatives and 400 as positives, while i have 1200 of each) and update \n"
	"   the data path to the current folder where all your stuff is in.\n"
	"\n"
	"opencv_traincascade -data C:\\Users\\Meyer\\Documents\\GitHub\\Object_Tracker_haas\\boxspeed -vec cropped.vec -bg negatives.txt -numPos 400 -numNeg 500 -numStages 20 -precalcValBufSize 1024  -precalcIdxBufSize 1024  -featureType HAAR -minHitRate 0.9 -maxFalseAlarmRate 0.5 -w 48 -h 48\n"
	"\n"
	"\n"
	"    This creates your classifier file called cascade.xml which you can then use with OpenCV haar detection. \n"
	"")
		
def main():
	var = raw_input('Choose one:'... 
		'\n-(1) Make positive training set by converting movie to images with original background'...
		'\n-(2) Make positive training set by converting images in folder to images with original background'...
		'\n-(3) Make positive training set by converting movie and add random background'...
		'\n-(4) Make positive training set by converting images in folder and add random background (can repeat)'...
		'\n-(5) Make negative training set by extracting BRIGHT background of images from images in folder (needs static BG)'...
		'\n-(6) Make negative training set by extracting DARK background of images from images in folder (needs static BG)'...
		'\n-(7) Make negative training set from from images in folder with no change to images'...
		'\n-(8) Convert movie to individual images (use this first if you want to use a movie for (5/6))'...
		'\n-(9) Create only the negatives.txt file from a given image folder'...
		'\n-(0) INSTRUCTIONS')
	
	
	if var is '1':
		print 'Converting movie with original background...'
		parse_images(use_movie = True, replace_ran_bg = False)

	elif var is '2':
		print 'Converting image folder with original background...'
		parse_images(use_movie = False, replace_ran_bg = False)

	elif var is '3':
		print 'Converting movie with random background...'
		parse_images(use_movie = True, replace_ran_bg = False)

	elif var is '4':
		print 'Converting image folder with random background...'
		parse_images(use_movie = False, replace_ran_bg = False)

	elif var is '5':
		print 'Extracting bright background only from folder...'
		create_empty_background_from_images_for_training(use_max = True, delta = 10)

	elif var is '6':
		print 'Extracting dark background only from folder...'
		create_empty_background_from_images_for_training(use_max = False, delta = 10)

	elif var is '7':
		print 'Converting images with no change...'
		convert_images_to_negatives_set(extract_only = True)

	elif var is '8':
		print 'Converting movie to individual frame images...'
		parse_movie_frames_to_file()

	elif var is '9':
		print 'Updating negatives.txt...'
		update_negatives_filename_list()

	elif var is '0':
		show_instructions()

			
if __name__ == "__main__":
	main()	


'''
guide and troubleshooting sources 
http://www.memememememememe.me/training-haar-cascades/
http://coding-guru.com/opencv-haar-cascade-classifier-training/
http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html
http://abhishek4273.com/2014/03/16/traincascade-and-car-detection-using-opencv/
https://github.com/wulfebw/mergevec/blob/master/mergevec.py
'''