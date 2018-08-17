# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/bar_chart.png "Bar Chart"
[image2]: ./writeup_imgs/orig_img.png "Original Imag"
[image3]: ./writeup_imgs/processed_img.png "Processed Image"

[image4]: ./new_signs_labeled/28.jpg "Road Work Sign"
[image5]: ./new_signs_labeled/13.jpg "Right-of-way At the Next Intersection Sign"
[image6]: ./new_signs_labeled/14.jpg "Children Crossing Sign"
[image7]: ./new_signs_labeled/25.jpg "Yield Sign"
[image8]: ./new_signs_labeled/11.jpg "Stop Sign"

[image9]: ./writeup_imgs/feature_map.png "Feature Map for Road Work Sign"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized images so that the pixel values are in between 0 and 1.  After normalization, I converted the color images to grayscale because training is faster with grayscale and based on experiments, color did not have any effect on the performance.  

Here is an example of a traffic sign image before and after processing.

![alt text][image2]
![alt text][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| 1st Layer:  Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| dropout: 50%												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| 2nd Layer:  Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					| dropout: 50%												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| 3rd Layer:  Fully connected		| outputs 120        									|
| RELU					| dropout: 50%												|
| 4th Layer:  Fully connected		| outputs 84        									|
| RELU					| dropout: 50%												|
| 5th Layer:  Fully connected		| outputs 43        									|
| Softmax				|        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:
* Optimizer:  Adam Optimizer
* Batch Size:  1024
* Number of Epochs:  150
* Learning Rate:  0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy on the data sets is located in the 6th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.989
* validation set accuracy of 0.941 
* test set accuracy of 0.927

For this project, I started with the well known LeNet-5 architecture.  It is a simple and a compact neural network, but suitable for traffic sign classification because traffis signs are relatively easy to classify due to their unique and distinct shapes and patterns.  In order to avoid overfitting, I implemented dropout technique.  This model is definitely working well for this task since training, validation, and test accuracy all show good accuracy.  From testing, I found 150 epochs was just around right number because after 150, accuracy on validation dataset stopped improving, but continued improving on training dataset, which was a sign of overfitting. 
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 
![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8]

The above images are after cropping and rescaling the original images to match the input size (32 x 32) of the neural network.
All images except the third one (stop sign) should be relatively easy to classify.  All those images have clear shape of the traffic signs and good contrast between the sign and the background.  The third image could be a little difficult since the sign almost takes the entire space of the image and it is angled as well.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children Crossing      		| Children Crossing   									| 
| Yield     			| Yield 										|
| Stop					| Stop											|
| Road Work	      		| Road Work					 				|
| Right-of-way At The Next Intersection			| Right-of-way At The Next Intersection     							|
 
The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. Since the test dataset accuray is 92.7%, this 100% is a bit better.  However, since this is only based on 5 signs, it's hard to derive a conclusion.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
For all images, the model is quite sure of the predictions and they are all correct.  For the third image, the model is slightly less confident with 84% prediction.

##### First Image (Children Crossing)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0 | Children Crossing | 
| .0  | Right-of-way at the next intersection |
| .0	 | Dangerous curve to the right|
| .0	 | Beware of ice/snow |
| .0	 | Bicycles crossing	|

##### Second Image (Yield)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0 | Yield | 
| .0  | Priority road |
| .0	 | Ahead only|
| .0	 | No passing|
| .0	 | No vehicles|

##### Third Image (Stop)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .84 | Stop | 
| .05  | Turn right ahead |
| .03 | Road work|
| .03	 | Keep right|
| .02	 | Speed limit (30km/h)|

##### Fourth Image (Road Work)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98 | Road work| 
| .01  | Wild animals crossing|
| .0	 | Double curve |
| .0	 | Road narrows on the right |
| .0	 | General caution	|

##### Fifth Image (Right-of-way At The Next Intersection)
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .92 | Right-of-way At The Next Intersection | 
| .07  | Beware of ice/snow |
| .0	 | Children crossing|
| .0	 | Slippery road |
| .0	 | Double curve|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

This is the feature map (1st activation layer) of the 1st image (Road Work).  The model is looking for edges in all directions and difference between the sign and the background.
![alt text][image9]


