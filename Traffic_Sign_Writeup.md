# **Traffic Sign Recognition** 

## Xiangjun Fan


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/distribution.png "Distribution"
[image4]: ./web_pics/Capture1.PNG "Traffic Sign 1"
[image5]: ./web_pics/Capture2.PNG "Traffic Sign 2"
[image6]: ./web_pics/Capture3.PNG "Traffic Sign 3"
[image7]: ./web_pics/Capture4.PNG "Traffic Sign 4"
[image8]: ./web_pics/Capture5.PNG "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 1024(32*32)
* The number of unique classes/labels in the data set is 43

#### 2. Here is an exploratory visualization of the train data set. It is a bar chart showing how the data is distributed on different traffic signs.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

I normalized the image data using (pixel - 128)/128 because normalization of x to (0, 1) circle will help NN find optimal solution more easily.


#### 2. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU			|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| dropout		| keep_prob 0.6									|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU			|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| dropout		| keep_prob 0.6									|
| flatten		| outputs 400									|
| Fully connected	| outputs 512  									|
| RELU			| 												|
| dropout		| keep_prob 0.6									|
| Fully connected	| outputs 128  									|
| RELU			| 												|
| dropout		| keep_prob 0.6									|
| Fully connected	| outputs 43  									|
| softmax		|												|
| <END>			|												|
 


#### 3. To train the model, I used an optimizer of "AdamOptimizer", the batch size of 256, number of epochs of 100, the learning rate of 0.001 and dropout keep_prob of 0.6.

#### 4. Describe the approach. 

My final model results were:
* training set accuracy of 0.9992
* validation set accuracy of 0.9701 
* test set accuracy of 0.9600

If an iterative approach was chosen:
* The first architecture that was tried was LeNet from previous course, because it works good on mnist data.
* The training and validation accuracy were not so good, less than 0.9, especailly validation accuracy was less than 0.8.
* The low training accuracy was strange, it meant the problem may be on the dataset itself. After a few rounds of debug, it turned out the normalization process was not correct, the data loaded in was unsigned int, therefore pixel-128 made it discreted, even I wanted the data to be centered to zero.
* After normalization is fixed, training accuracy is approaching 0.9999..., but validation accuracy is just above 0.9. It indicates needing of regularization. So dropout was added to LeNet. 
* Batch size, dropout prob, CNN/FC size were tuned to find the best solution.
 

### Test a Model on New Images

#### 1. Here are five German traffic signs that I found on the web:

![web pic1][image4] ![web pic2][image5] ![web pic3][image6] 
![web pic4][image7] ![web pic5][image8]


#### 2. Here are the results of the prediction:

| Image			        	|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| General caution  		| General caution								|
| Wild animals crossing		| Wild animals crossing							|
| Road narrows on the right	| Road narrows on the right						| 
| Slippery road	      		| Slippery road					 				|
| Priority road			| Yield      									|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96%.
The wrong image was from a different source, and it was a Priority road sign with some dirty drawing on it. I intended to include this picture to see how robust the model is. It turned out for image with a different color pattern, and partially covered, the trained model did not work well. The solution could be to convert image to grayscale before training, and do augumentation on images to create more data sample.

#### 3. The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
The top five soft max probabilities were

For the first image:
web_pic: General caution
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 1.0000000 | General caution |
| 0.0000000 | Traffic signals |
| 0.0000000 | Speed limit (20km/h) |
| 0.0000000 | Speed limit (30km/h) |
| 0.0000000 | Speed limit (50km/h) |

For the second image:
web_pic: wild animals crossing
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 1.0000000 | Wild animals crossing |
| 0.0000000 | Double curve |
| 0.0000000 | General caution |
| 0.0000000 | Slippery road |
| 0.0000000 | Dangerous curve to the left |

For the third image:
web_pic: road narrows on the right
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 0.9977132 | Road narrows on the right |
| 0.0016438 | General caution |
| 0.0006019 | Pedestrians |
| 0.0000292 | Bicycles crossing |
| 0.0000087 | Right-of-way at the next intersection |

For the forth image:
web_pic: slippery road

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 0.9999714 | Slippery road |
| 0.0000286 | Dangerous curve to the left |
| 0.0000000 | Bicycles crossing |
| 0.0000000 | Road narrows on the right |
| 0.0000000 | Dangerous curve to the right |

For the fifth image:
web_pic: priority road

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| 0.5161353 | Yield |
| 0.3686315 | Priority road |
| 0.0820100 | No entry |
| 0.0063870 | Speed limit (30km/h) |
| 0.0063788 | Stop |

