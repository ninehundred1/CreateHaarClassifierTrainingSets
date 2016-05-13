
# Create image sets to train your own Haar Classifiers

##### Automatically create training sets of 1000s of images with random backgrounds where the position of your object is automatically marked to be used by OpenCV training algorithms. 
#

OpenCV documentation is sometimes lacking, so this is trying to help people out that want to train their own classifiers but might not know where to start.

There are a few ways to do that. 


**1.You have a nice static movie**

If you have a movie where the background is not moving (like a fixed camera) as here: 
![OBJECTTRACKER_allobjects](http://i.imgur.com/JVZu6Wl.jpg)

you can use this module to just extract the backgrounds only:
![OBJECTTRACKER_allobjects](http://i.imgur.com/faxQP0G.jpg)

and nicely mark your objects of tracking in 1000s of images from the movie:
![OBJECTTRACKER_allobjects](http://i.imgur.com/eTiGWbc.jpg)






[*See this repository for some more background details*](https://github.com/ninehundred1/Object_Track_Haar_classifier "CV2")


**2.You don't have a nice static movie**

In that case, you need to go and start to get your own images to use. You can start creating your own image sets by placing your object on a white paper taking pictures from different angles, or asking your friend to pose in front of a white wall.


If that is not possible, or you are too lazy, go to Google and look for images with a nice clean background. Easiest is to use images with the same background, white for example.

As in this recent project, just googling a keyword often gets you nice images with white backgrounds:

![OBJECTTRACKER_allobjects](http://i.imgur.com/AVb8xy6.jpg)

Take as many images you want (you probably want around 50 or more), and then ideally crop them in a way that you are only left with a single object per image on a white background. 
Depending what you want to train, it might make sense to get the object from different angles and colors.


**Get your random backgrounds**

The other thing you need (in addition to your images containing your object of desire) is many random photos. You want to train your classifiers to what it should detect (positive set), and also what it should not detect (negative set), so the more varied random images you have, the better. 

Also, if you don't have a movie and only managed to find 50 images of your object, using those in random background gets you more to train with.

The module will also slightly rotate and distort your objects randomly, which will further make many (1000s) out of few (50) original images that are all a little different.

**Here is what this module does:**

1. In the case of using a movie as the image source and replacing random backgrounds, these are the steps that are done from the initial iamges to the final image:



![OBJECTTRACKER_allobjects](https://camo.githubusercontent.com/850b0d35defe2ace5c323700f2b3fac219e318aa/687474703a2f2f692e696d6775722e636f6d2f67516b56464a412e6a7067)



In addition to the images a txt file with the name of the image and the number and position of the your object will also be generated:


    positive_images/frame0.jpg	1	149	21	315	355
    positive_images/frame1.jpg	1	152	1	359	366
    positive_images/frame2.jpg	1	1	168	302	231



If you are not using a movie, you need to supply a suitable image for the background substraction. In the case of the images from the google search above, just a plain white image is enough.

When done you can use the training sets to train your own classifiers (see below for instructions).

![OBJECTTRACKER_allobjects](https://camo.githubusercontent.com/b3f49f2df5541357b8cdbfcdcfdc30618da7999a/687474703a2f2f692e696d6775722e636f6d2f6e684a7949634a2e676966)



## Instructions

WHAT YOU NEED TO GENERATE HAAR CLASSIFIER TRAINING SETS:

A. A folder of OBJECT IMAGES that has your target object over the same white/black/etc background, 
   or a movie where the background remains constant (doesn't move) or has the same color throughout.
   These images are for your positives training set.
   You probably should have around 1000-2000 of those images (if not using a movie). 
   If you only have a few (20 or 50) you should run the option where you replace the background with 
   random images (4) and run it a few times.
   Each time you run it the images come out different, so you will end up with different images made
   from the same few initial images.

B.  You also need one BACKGROUND IMAGE that is only the background with ideally the same size as other
   images. This image is used to automatically find the borders of your object which is required for 
   training, so the algorithm knows where your object of interest is.
   If you have images of your object over white background, one all-white image is fine. For a movie with a            stationary background or color, you can create clean images of the background using option 5 or 6.
  

C. A folder with RANDOM IMAGES to form your negatives training set. You can use photos from your phone, 
   etc. Again, try to have 1000-2000.
   If you use the option to relace the background behind your object (mentioned in A), you need two sets
   of image, each around 1000-2000 images.




**To generate the training data sets for the Haar Classifiers from the images:**


Make Positives set:

a) If you have a folder of a few images (50 or so) with your object over a smooth background (eg white or black etc)
use option 4 to create your positive training set. You will also need to supply one image of just the 
background (same color as your background) and a folder that has many random images.
You will then be asked how often to cycle through those object images. Choose it to get you to 1000-2000 
images total.

b) If you have a folder with a lot images and the same background (such as extracted from a movie file), select
option 2. This will not replace the background but only mark the object. For that, you need again supply an 
image with just the background for extraction.

c) If you have a movie on a steady background (some kind of fixed camera), first create a background image out
of the movie using option 5 or 6 (which will create a negatives set also, which you might not need),
depending if your background is brighter than your object or darker.
This will create a few images with just the background. Then you have the option to create a positives training
set straight out of the movie by using option 1 or by replacing the background using option 3. 
Use one of the background images generated before as a background image. If using option 3, then select a
folder that contains a lot of random images).



Make Negatives set:

d) If you have a movie with a steady background you can use option 5 or 6 to create a set of images of just
the background. This might not be ideal as there is very little variance between the images, which is not
what you really want (you want to train the classifier on many things that are not your object, not few).

e) Ideally you just use a lot of random images, as that tells the classifiers many things that are not 
what to look for. Use option 7 to create a training set out of a folder with many random images.
Use 1000-2000 random images in one folder.


Other options:
You can extract a movie into individual jpg frames by using option 8. You can then use those for b),
if you are only interested in certain frames of a movie.

If you want to use random images for your negatives set in e), and later want to add more images, you can just
create a new txt file that includes all images in the folder by using option 9.





**USING THE TRAINING SETS**

1. copy opencv_createsamples.exe and opencv_traincascade from the opencv folder you dowloaded initially
   (check folder C:\Users\Meyer\Downloads\opencv\build\x64\vc10\bin)

   I also copied all dll files from that folder into the same directory as well to the same directory
   with the folders positive_images and negative_images. Those two folders also contain the files 
   positives.txt and negatives.txt, which you need to move one folder up.

   So in your working folder you should now have:
		Folder negative_images
		Folder positive_images
		Txt file negatives.txt
		Txt file positives.txt
		opencv_createsamples.exe 
		opencv_traincascade
		(and if error the other files from the bin folder)


2. Create a vector file of the images in the windows command window.
   While in the above directory run (but change num to your image number of positve images):


    opencv_createsamples -info positives.txt -bg negatives.txt -vec cropped.vec -num 1279 -w 48 -h 48

  This creates a vector file called cropped.vec


3. Train the cascade using above vector file (can take hours to days). 
   Again on command while in the same directory run below, but change the number of images to less 
   what you have (I use 500 as negatives and 400 as positives, while i have 1200 of each) and update 
   the data path to the current folder where all your stuff is in.


    opencv_traincascade -data C:\Users\Meyer\Documents\GitHub\Object_Tracker_haas\boxspeed -vec cropped.vec -bg negatives.txt -numPos 400 -numNeg 500 -numStages 20 -precalcValBufSize 1024  -precalcIdxBufSize 1024  -featureType HAAR -minHitRate 0.9 -maxFalseAlarmRate 0.5 -w 48 -h 48


This creates your classifier file called cascade.xml which you can then use with OpenCV haar detection. 



### To run you need python 2.7 and the following packages:

- cv2
- numpy
- Tkinter


emails to:
- <fuschro@gmail.com>