# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./9.jpg "No passing"
[image2]: ./18.jpg "General caution"
[image3]: ./20.jpg "Dangerous curve to the right"
[image4]: ./31.jpg "Wild animals crossing"
[image5]: ./My_Signs/dangerous_curve_right.jpg "dangerous_curve_right"
[image6]: ./My_Signs/kmph30.jpg "Speed limit 30"
[image7]: ./My_Signs/kmph70.jpg "Speed limit 70"
[image8]: ./My_Signs/no.jpg "No entry"
[image9]: ./My_Signs/stop.jpg "Stop Sign"
[image10]: ./My_Signs/straight.jpeg "Straight only"
[image11]: ./Bar_Chart/bar_dangerous_curve_right.jpg "dangerous_curve_right"
[image12]: ./Bar_Chart/bar_kmph30.jpg "Speed limit 30"
[image13]: ./Bar_Chart/bar_kmph70.jpg "Speed limit 70"
[image14]: ./Bar_Chart/bar_no.jpg "No entry"
[image15]: ./Bar_Chart/bar_stop.jpg "Stop Sign"
[image16]: ./Bar_Chart/bar_straight.jpeg "Straight only"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/sinyyl/Deeplearning_Traffic_Sign.git)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Here are 4 randomly picked traffic sign pictures in the dataset.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did not convert the images to gray scale which might be the conventional way, because I think the color in the traffic sign actually contains quite a lot of the information.

As a mandatory step, I normalized the data from between 0 and 255 to between -1 to 1. Therefore, they can have the mean 0 and equal variance.

However, the suggested way of '(pixel - 128)/ 128' was not going smoothly. It produced the result between 0 and 1. I did not notice the program until I realized no matter how I tune the model, it is still not producing acceptable accuracy. It was solved by converting the data type from int to float. 




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 28*28*6 	|
| RELU					|												|
| Avg pooling	      	| 2x2 stride, valid padding, outputs 14x14x6  	|
| Convolution 1x1		| 1x1 stride, valid padding, outputs 10*10*16	|
| Max pooling			| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flatten				| faltten to 1 dimension  1 * 400				|
| Fully connected		|outputs 1 * 84									|
| Fully connected		|outputs 1 * 43									|
| Droupout		 		|Keep rate 0.5									|
| Output				|outputs 1 * 43									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For the optimizer, I used AdamOptimizer which is also used in the previous exercise. To increase the accuracy, I used a low learning rate of 0.0003 with a large epoch of 220. The batch size is set to a standard of 128.
At first, when I train the model, I realize the model is overfitted as the trainning accuracy is 1 and the validation accuracy is still only about 0.9. So I used the technique of drop out with the keep rate tuned to 0.5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999.
* validation set accuracy of 0.934.
* test set accuracy of 0.928.

The architecture is LeNet which is a well developped architecture. It is optimized for processing images. LetNet can process most type of images which also inclues classify traffic signs. It was also proven to work pretty good on the previous exercise.
However, for it to have an validation accuracy of 0.93 and up, there are still some tunning need to be done. I added a drop just before the output layer to prevent overfitting. 
In the beginning stage, I changed the 2 max pooling with average pooling to preserve more information. But, the model tend to overfit still. After many runs,I setteled with only the second pooling changed to avg pooling and used keep rate for drop as 0.5 to get a balance between overfit and underfit.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]

The last image might be difficult to classify because the image might be different from the ones in the dataset.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	 	 						| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30 kmph	| Speed Limit 30 kmph 							| 
| Speed Limit 70 kmph	| Speed Limit 70 kmph							|
| No entry				| No entry										|
| Stop Sign				| Stop Sign					 					|
| Straight only			| Straight only									|
| Dangerous curve to the right| Stop Sign								|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. The accuracy is lower than the 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.


For the first image, the prodiction is overwhalmingly toward Road work as shown in the figure below, however it is actually Dangerous curve to the right.
![alt text][image11] 

All the predictions have a probablity very close to 1. The probablity for other predictions can be ignored as shown in the bar chart. This is the prediction for speed limit of 30 kmph, which correctly predicted.
![alt text][image12] 

My model correctly predicted it as speed limit of 70kmph. It has a probablity very close to 1.
![alt text][image13] 

It correctly predicted the image as no entry with a almost 100% probablity. 
![alt text][image14] 

A very close to 1 prediction of stop sign which is correct.
![alt text][image15] 

Correct prediction of straight only with very close to 1 prediction.
![alt text][image16]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

## LICENSE
[MIT LICENSE](./LICENSE)
