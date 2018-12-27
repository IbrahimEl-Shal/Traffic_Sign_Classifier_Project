# **Traffic Sign Recognition**

## Udacity Self Driving Car nano-degree: Project 3

### A Convolutional Neural Network traffic sign classifier using deep learning techniques

---

The goals / steps of this project are the following:
* Import the needed Packages
* Load the data set
* Explore, summarize and visualize the data set
* Pre-process the Data Set (normalization, grayscale, etc.)
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/Sign_Traffic.png "Visualization"
[image2]: ./output_images/preprocessing_1.png "Distribution"
[image3]: ./output_images/preprocessing_2.png "Preprocessing"
[image4]: ./output_images/accuracy.png "Accuracy"


---
### Data Set Summary & Exploration

#### 1. The statistical information about the dataset was calculated using numpy
* The size of training set is 31367 images.
* The size of the validation set is 7842 images.
* The size of test set is 12630 images.
* The shape of a traffic sign image is 32 x 32 x 3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

At first, I plotted some 20 random images from the training set and here it is:

![alt text][image1]

### The Model Architecture

#### 1. Preprocessing
Firstly converting the RGB images to grayscale because what we're trying to predict is the shape of the sign and converting to grayscale will change the dimensions of the image and make it 1-dimensional instead of 3-dimensional (for 3 colors), and that will speed up the training. After converting to grayscale, all the images were normalized using the formula: pixel = (pixel - 128) / 128. 
Normalization is a very important step in speeding up the training and in reducing the chances of getting stuck at a local optima.

Here's a visualization of an image getting through the preprocessing steps:

![alt text][image2]

![alt text][image3]

#### 2. The LeNet architecture

The model I used was the LeNet model that we learned in class. It consisted of the following layers:

| Layer         		|     	Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|


#### 3. Adam Optimizer

To train the model, I used tensorflow's Adam Optimizer. The Adam optimization method was published in a famous research paper by Diederik P. Kingma, Jimmy Ba, titled "Adam: A Method for Stochastic Optimization". It's as the - quoting the asbtract - "an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments."

#### 4. Approach for finding a solution: Why LeNet was chosen?

As mentioned above, the architecture chosen for this model was the famous LeNet architecture. It was initially made for classifying handwritten numbers. The architecture took the inputs as images, and pretty much depended on the same factors as if it was a sign classifier. Whether it was focusing on the shape and not the color, or other factors that shaped how the model work.
For just LeNet it had some drawbacks. The model's final accuracy wasn't amazing, but it was pretty descent for a "plug and play" architecture.

Wrapping up, My final model results were:
* validation set accuracy of 95.6%
* test set accuracy of 93.5%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction      	|
|:---------------------:|:---------------------:|
| 20 km/h         	   	| Right Only   			|
| General Caution     	| General Caution 		|
| Bumpy Road			| Bumpy Road			|
| Right Only	      	| Right Only	 		|
| Stop          		| 30 km/h       		|
| Stop                  | Stop                  |

The model was able to correctly guess 4 of the 6 traffic signs, which gives an approximate accuracy of 66.6%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For most of the images, the model was certain of it's predictions. The top five soft max probabilities were

| Probability         	|     Prediction 		|
|:---------------------:|:---------------------:|
| 1.0         		    | 20 km/h   	   	   	|
| 1.0     			    | General Caution		|
| 1.0			        | Bumpy Road			|
| 1.0	      	      	| Right Only			|
| 0.98				    | Stop      			|
| 1.0                   | Stop                  |
