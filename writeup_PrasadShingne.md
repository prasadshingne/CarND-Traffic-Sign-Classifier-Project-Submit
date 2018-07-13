# **Traffic Sign Recognition** 

## Writeup

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

[image1]:  ./Figures/Example1.jpg "Visualization"
[image2]:  ./Figures/histogram.jpg "Histogram"
[image3]:  ./Figures/ColorVSGray.jpg "Color VS Gray"
[image4]:  ./Figures/OriginalVSNorm.jpg "Original VS Normalized"
[image5]:  ./Figures/MyPictures.jpg "My Pictures"
[image6]:  ./Figures/MyGuesses.jpg "My Guesses"
[image7]:  ./Figures/Conv1Features.jpg "Conv1Features"
[image8]:  ./Figures/Conv1ReluFeatures.jpg "Conv1ReluFeatures"
[image9]:  ./Figures/Conv1Pool.jpg "Conv1Pool"
[image10]: ./Figures/Conv2Fatures.jpg "Conv2Fatures"
[image11]: ./Figures/Conv2ReluFeatures.jpg "Conv2ReluFeatures"
[image12]: ./Figures/Conv2Pool.jpg "Conv2Pool"



## Rubric Points (https://review.udacity.com/#!/rubrics/481/view)
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Link to my [project code](https://github.com/prasadshingne/CarND-Traffic-Sign-Classifier-Project-Submit/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic [cell 2]
signs data set:

* Number of training examples     = 34799
* Number of validation examples   = 4410
* Number of testing examples      = 12630
* Image data shape                = (32, 32, 3)
* Number of unique labels/classes = 43

#### 2. Include an exploratory visualization of the dataset.

The figure below displays 10 randomly selected images from the training set along with the label. The sign for 50 km/h (labeled as 2) is repeated in this random sample. [cell 3]
![alt text][image1]
The next figure shows the distribution of the traffic signal labels from the training dataset in the form of a histogram. The occurrence of labels between 2 and 15 is higher than the rest of the set. [cell 4]
![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I converted the data to grayscale. This is because the three color channels do not add to significant information as input to the CNN. [cell 5]
The next figure shows 40 figures with and without grayscaling. In both cases the signs can be clearly identified. [cell 6] 
![alt text][image3]
Then I normalized the grayscaled images - this brings the mean of the data close to 0. [cell 8 and cell 9] Literature says that normalization makes training easier and speeds it up.

There is very little difference apparent between the original image and a normalized image [cell 10]:

![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model was based on LeNet architecture [cell 14]. I added dropout layers after the RELU in the fully connected layers in order to avoid overfitting during training. The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, output 28x38x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Fully connected		| 400 (input) to 120 (output)						|
| RELU					|												|
| Dropout        		| Randomly dropout some units during training 			|
| Fully connected		| 120 (input) to 84 (output)						|
| RELU					|												|
| Dropout        		| Randomly dropout some units during training 			|
| Final Fully connected		| 84 (input) to 43 (output)						|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained using the "AdamOptimizer" [cell 16] with a batch size = 100 and epochs = 75 [cell 13], and a learning rate of 0.0006 [cell 16].

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were [cell 20]:
* training set accuracy of 0.999
* validation set accuracy of 0.967 
* test set accuracy of 0.943

Some notes about the iterative changes to the architecture and tuning parameters for epochs, learn rate and batch size:
* I started exactly with the LeNet architecture provided in the class material. This did not work as well and I added the two dropouts after the fully connected layers to avoid overfitting. A 50% dropout probability worked the best.
* Concequently I had to increse the number of epochs to 75 which seemed to give the best results.
* I played around with the batch size a little and settled on 100 as it gave the best results.
* The learn rate had significant effect on the training. A value of 0.0006 gave the best balance of learning fast enough without jumping around the optimum too much.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image5] 

The slippery road [first] and road work [sixth] image might be difficult to classify because they simply do not appear to be clear. The road work image seems to be taken at an angle and the sign appears squeezed horizontally. The first, second, third, fifth and seventh images also have significant "noise" (trees and other background information) behind the sign of interest which could make the classification difficult.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image6] 

The model was able to correctly classify all the selected test images [cell 24 and cell 25].

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 24th and 26th cell of the Ipython notebook.

The model picks the correct label with very high probability (more than 5 decimal places) for all the images. The model predicts a wrong label for the third image (30 km/hr) with a probability of 0.00001. 

The table below presents the top 5 softmax probabilities for each image along with the corresponding labels.

Predictions:
* 1.0       0.       0.       0.       0.     
* 1.0       0.       0.       0.       0.     
* 0.99999   0.00001  0.       0.       0.     
* 1.0       0.       0.       0.       0.     
* 1.0       0.       0.       0.       0.     
* 1.0       0.       0.       0.       0.     
* 1.0       0.       0.       0.       0.     
* 1.0       0.       0.       0.       0.     

Labels:

* 23 20 30 11 28
* 18 27 26 11 37
* 1  0  2   4 29
* 38 13 34 25  3
* 17 14  8  4 15
* 25 22 26 24 20
* 13 38 10  9 14
* 33 13 35 12 14



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Cells 28 to 33 shows the visualization of my nural network through conv1, conv1_relu, conv1_pool, conv2, conv2_relu and conv2_pool. Conv2 onwards it becomes difficult to intuitively make sence of the features displayed. 

The next three figures show the conv1, conv1_relu and conv1_pool visualization respectively - 

![alt text][image7] 
![alt text][image8] 
![alt text][image9] 

The NN identifies lines and blobs that look like the boundaries of the sign and the features displayed within. This helps the NN identify the different parts of each sign and put them together to identify the sign.
