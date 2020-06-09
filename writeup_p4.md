# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/case2open.png "case: in open area"
[image2]: ./output/case1brige.png "case: on brige"
[image3]: ./output/center_demo.jpg "center image"
[image4]: ./output/right_recovery.jpg  "right Recovery Image"
[image5]: ./output/left_recovery.jpg  "left Recovery Image"
[image6]: ./output/center_recovery.jpg  "center Recovery Image"
[image7]: ./output/success2.png "loss trend"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_p4.md summarizing the results
* helper.py containing the data_process functions to collect data

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
>NOTE: This model is trained and tested in my local machine wiht GPU:GTX1060 6G version. For me, I don't have a super-fast network to upload the images data. So I choose to train my model on local machine. And it works well.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is just same as Lenet-5 in p3 project except the normalization input layer.

The model includes RELU layers to introduce nonlinearity (code line 58 and 63), and the data is normalized in the model using a Keras lambda layer (code line 54 and 55). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers(with keep_prob = 0.5) in order to reduce overfitting (model.py lines 67). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 116-120). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
Also I used the **ModelCheckpoint**(model.py line 104) and **stopper**(model.py line 105)  modules to save best model and use early stopper to avoid overfiting.

#### 3. Model parameter tuning

The model used an adam optimizer. (model.py line 102).
I found when the epoch increased , the loss decreased slowly so I google a way to
auto-ture the learning rate by line 106 in model.py.


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also build a filter function in `helper.py` named **filter_samples()** to cut off some zero steering data to avoid the model "learn too much to do nothing".
I also used the center\left\right images and their flip images to generate enought data to train the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to predict a proper steering angle considering current car positon on the road. 

My first step was to use a convolution neural network model similar to the lenet-5. I thought this model might be appropriate because this net architect is quite simple and easy to revise.It works well for classfing the image which means it's good at recognising the chracters in image. 


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I add up a drop out layer. I also found this layer not only help avoiding overfitting but also controling the car smoothly.

Then I think the normaliaztion layer maybe helpful, but it's proved that when add the nomalization layer the model behavior became unsteady.

The final step was to run the simulator to see how well the car was driving around track one. 
* case 1:
   
There were a few spots where the vehicle fell off the track especially when the road is open aera. The car intend to drive into the open aera. I think it because the image is not similar to ohter road line. It's quite blur. 
The case 1 is shown in picture below:
![alt text][image1]
To improve the driving behavior in these cases, I tried to drving closing to the right line and recorey from it more frequently.
* case 2:
When on the brige, the car tend to run on one side of the brige. I think this is because when I drived on the brige manually, I nearly didn't steer the car at all(the car's derection is already good).
To improve the driving behavior in these case, I mean to steering the car on brige more ofen so the car knows what to do with this situation.

*collect more data
when try collect more data with:
1. two or three laps of center lane driving
2. one lap of recovery driving from the sides
3. one lap focusing on driving smoothly around curves
the loss is sharply down.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road as shown in vedio `/output/success1.mp4`


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional 

With model.summary(), the architect of the lenet-5 model is shown below:
| layer(type)| output shape | praram num |
|:-----|:--------|:------:|
| lambda_1  | (None, 160, 320, 3) | 0 |
| cropping2d_1   | (None, 60, 320, 3) | 0 |
| conv2d_1  | (None, 56, 316, 6) | 456 | 
| activation_1    | (None, 56, 316, 6) | 0 | 
| max_pooling2d_1    | (None, 56, 316, 6) | 0 | 
| conv2d_2 (Conv2D)     | (None, 24, 154, 16)| 2416 | 
| activation_2    | (None, 24, 154, 16) | 0 | 
| max_pooling2d_2    | (None, 12, 77, 16)  | 0 |
| flatten_1 (Flatten)   |  (None, 14784)    | 0 |
| dense_1 (Dense)    |  (None, 120)  | 177420|
| dense_2 (Dense)    |  (None, 84)  | 10164|
| dense_3 (Dense)    |  (None, 1)  | 85|
_________________________________________________________________
Total params: 1,787,409
Trainable params: 1,787,365
Non-trainable params: 44
_________________________________________________________________

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery from side lines. These images show what a recovery looks like starting from road lane sides:

![alt text][image4]
![alt text][image5]
![alt text][image6]

Then I repeated this to run some data in anticlockwise direction.

To augment the data sat, I also flipped images and angles thinking that this would add more data. 

While generating more data, I also improve the data feeding way.
* At first, I use the `pickle` module to store the processed data. It work well at the first place, but as the data size increasing, the local machine's memeory is not enough.
* Then I tried the `generator` technique to deal with huge data.

While I have more data, but I found the model's behavior is not steady as expected. I searched in the 'Knowledge' forum, I think filtering some zero steering angle image data(not all of them) maybe helpful. And with this operation ,the model is more robust.


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by runing the model. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here's a exsample of how the loss in training set and validation set.
![alt text][image7]

### What to explore next
* generate more high quality data
I think the input data is absoutly important, the model will learn what you feed it.
* try more powerful network
Although I have tried the nivida's end to end learning CNN net work, it didn't work well I think I'll explore more model with more time to understand how to design the architecure of a useful network.