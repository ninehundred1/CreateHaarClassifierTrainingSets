
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

In the case of using a movie as the image source with a static background a few images get averaged to remove the object and create a background-only image. That background is then substracted from all images to extract the object only. Then the image is thresholded, contoured to selectively keep the object. The background is then replaced by a random image. A vector of the border of the object is saved along with the file name and that is used with the images as the training set.


![OBJECTTRACKER_allobjects](https://camo.githubusercontent.com/850b0d35defe2ace5c323700f2b3fac219e318aa/687474703a2f2f692e696d6775722e636f6d2f67516b56464a412e6a7067)



In addition to the images a txt file with the name of the image and the number and position of the your object will also be generated:


    positive_images/frame0.jpg  1 149 21  315 355
    positive_images/frame1.jpg  1 152 1 359 366
    positive_images/frame2.jpg  1 1 168 302 231



If you are not using a movie, you need to supply a suitable image for the background substraction. In the case of the images from the google search above, just a plain white image is enough. 
As before the image gets thresholded, the contour of the object is found and the background replaced.

The images created this way are used as positive sets, then random images without your object are used negative sets and both sets are converted to vectors, which are then used in training to create the classifier file.

That file can then be used with CV2 Haar detection (below the left is using trained Haar classifiers, the right is using histogram shifting without training).


![OBJECTTRACKER_allobjects](http://i.imgur.com/0KSFkTG.jpg)



When done you can use the training sets to train your own classifiers (see below for instructions).

![OBJECTTRACKER_allobjects](https://camo.githubusercontent.com/b3f49f2df5541357b8cdbfcdcfdc30618da7999a/687474703a2f2f692e696d6775722e636f6d2f6e684a7949634a2e676966)



# Instructions

### 1. GET YOUR IMAGES

**A. A folder of OBJECT IMAGES or a source movie**

As the goal here is to locate the object automatically, the images should be over the same background (say all images over white), or a movie where the background remains constant (doesn't move) or has the same color throughout.

   These images are for your positives training set.
   You probably should have around 1000-2000 of those images (if not using a movie). 
   If you only have a few (20 or 50) you should run the option where you replace the background with 
   random images (4) and run it a few times.
   Each time you run it the images come out different, so you will end up with different images made
   from the same few initial images.
   
   All images should be the same size ideally, so you can use photoshop GIMP or 
[ImageJ](https://imagej.nih.gov/ij/download.html "CV2") to crop them correctly.

*Here an example of some images (I have around 80 or those):*


![folder](http://i.imgur.com/UDyi5p4.jpg)

**B.  One BACKGROUND IMAGE**

Again we plan to extract the object of interest, so easiest to do that is by substracting the background, so that is what this image is for. Ideally the image is the same size as other images. If you use photoshop etc to crop the source image, just create a new image with the color of the background and the size of the other source images.
For example,if you have images of your object over white background, one all-white image is fine.

For a movie with a stationary background or color, you can create clean images of the background using option 5 or 6, which should be usable as a background image.
   
 

**C. A folder with RANDOM IMAGES**

These form your negatives training set. You can use photos from your phone,etc. Again, try to have 1000-2000. They don't need to be the same size, and also not the same size as the other images from above.


If you use the option to relace the background behind your object (mentioned in A), you need two sets of image, each around 1000-2000 images. One set is used to replace the background, the other is used here as the negative set. Don't use the same images for both.



### 2. GENERATE YOUR TRAINING SETS

#### Make Positives set

In the windows command prompt run the file (make sure you are in your virtual environment with CV2, etc (see below for dependencies) installed.

    python create_haar_training_sets.py



##### A) If you have a folder of a few images (50 or so) with your object over a smooth background (eg white or black etc)
use **option 4** to create your positive training set. 

a. You will be promted to select the folder with the source images first (the folder that has all your objects over the same background).
**It will tell you how many images it loaded, take note of that number so you can set the times of re-use accordingly.**

b. Then select the background image (eg the all white image).


c. Then select the folder with the random images (remember you need two folders with random images, the second you will use later).

Now it tell you how many files are in the folder, then it will ask you how often you want to run the images. With the number you remembered from above (the number of pictures you had in your source image folder, in my case 80), now set the time you want to re-use the images. I have 80, and I want around 1400, so I will run it 18 times.

**A new folder calles positive_images will be created in the same directory where your background image was in.**

Now you have plenty of images. Note that the images generated do not include the bounding box as shown in the example above. 

**These images are the ones used directly for training, and the bounding box is stored in the also generated file positives.txt**


##### Alternative without background replacement
If you have a folder with a lot images and the same background (such as extracted from a movie file, which you can do using **option 8**), select **option 2** to create your positive data set. This will not replace the background but only mark the object. Again, as the marking of the object functions automatically, you need to substract a background. This is why using images with changing backgrounds won't work. So again, you need to supply an image with just the background for extraction. If you use this, there is the option to run it several times, and as there is a random amount of distortion and rotation, running the images several times will get you a larger data set.


To use a movie without background replacement, choose **option 1**, while again supplying an image of the background for automated object extraction.

##### B) If you have a movie file with a steady background
First create your *background image*. If you have a movie on a steady background (some kind of fixed camera), first create a background image out of the movie using **option 5 or 6**.
This will create many images, all not containing your object. You can also use these images as your negative image set, though the lack of variability makes it less ideal.
Depending if your background is brighter than your object or darker than your object you have two options (5 or 6).


Next you have the option to create a positives training
set straight out of the movie by using **option 1** or by replacing the background using **option 3**.

Use one of the background images generated before as a background image i5f using option 3, then select a folder that contains a lot of random images.



#### Make Negatives set

If you have a movie with a steady background you can use **option 5 or 6** to create a set of images of just the background. This might not be ideal as there is very little variance between the images, which is not
what you really want (you want to train the classifier on many things that are not your object, not few).

Ideally you just use a lot of random images, as that tells the classifiers many things that are not what to look for. Use **option 7** to create a training set out of a folder with many random images.
Use 1000-2000 random images in one folder.


### Other options:
You can extract a movie into individual jpg frames by using **option 8**. You can then use those for b),if you are only interested in certain frames of a movie.

If you want to use random images for your negatives set and later want to add more images, you can just
create a new txt file that includes all images in the folder by using **option 9**.





### 2. CREATE CLASSIFIERS FROM TRAINING SETS

##### A. Copy *opencv_createsamples.exe* and *opencv_traincascade* from your OpenCV source folder into your current folder
this is my folder: C:\Users\Meyer\Downloads\opencv\build\x64\vc10\bin

   I also copied all dll files from that folder into the same directory as well to the same directory
   with the folders *positive_images* and *negative_images*. Those two folders also contain the files *positives.txt* and *negatives.txt*, which you need to move one folder up.

   So in your working folder you should now have:
   
    Folder negative_images
    Folder positive_images
    Txt file negatives.txt
    Txt file positives.txt
    opencv_createsamples.exe 
    opencv_traincascade
    (and if error the other files from the bin folder)


##### A. Create a vector file of the images in the windows command window.
   While in the above directory run within your normal windows command prompt (but change num in the command below to your image number of positive images):


    opencv_createsamples -info positives.txt -bg negatives.txt -vec cropped.vec -num 1353 -w 48 -h 48

  This creates a vector file of your images called *cropped.vec*


##### C. Train the cascade using above vector file (can take hours to days). 
   Again on command while in the same directory run below, but change the number of images to less what you have (I use 500 as negatives and 400 as positives, while i have 1200 of each) and update the data path to the current folder where all your stuff is in.
 


    opencv_traincascade -data C:\Users\Meyer\Documents\GitHub\Object_Tracker_haas\boxspeed -vec cropped.vec -bg negatives.txt -numPos 700 -numNeg 900 -numStages 20 -precalcValBufSize 1024  -precalcIdxBufSize 1024  -featureType HAAR -minHitRate 0.9 -maxFalseAlarmRate 0.5 -w 48 -h 48


This creates your classifier file called *cascade.xml* which you can then use with OpenCV haar detection. 

If you get errors lower the numbers you used in above command for negatives and positives.


### 3. Using your new classifier file

You can now use the new *cascade.xml* file in CV2's function

    cv2.CascadeClassifier()



[If you look at the file Track.Class_haar.py here, ](https://github.com/ninehundred1/Object_Track_Haar_classifier/blob/master/TrackClass_haar.py "CV2") you can check the function below on how to use it (I just renamed the file *cascade.xml* to *haar_cascade_Mice_randombg.xml* and moved it into the same folder the .py file is in.

    def check_object_position(self, termination, frame_count, object_updates_per_frame, area_updates_per_frame, current_object_update_offset, logger):
    logger.debug("Offset in:"+str(current_object_update_offset))
    objects_in_areas = []
    current_object_tracked = 0
    current_target_checked = 0
    offset_for_object = 0
    offset_for_target = 0
    use_haar = True

    cascade = cv2.CascadeClassifier("haar_cascade_Mice_randombg.xml")



### To run you need python 2.7 and the following packages:

- cv2
- numpy
- Tkinter


emails to:
- <fuschro@gmail.com>