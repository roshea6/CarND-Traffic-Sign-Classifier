# **Traffic Sign Recognition** 

Ryan O'Shea

I exported the notebook as an html file and uploaded to the workspace as recommended. The file is called Traffic_Sign_Classifier.html


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[results]: ./ex_imgs/5_img_result.PNG "Results"
[softmax_results]: ./ex_imgs/5_img_softmax.PNG "Softmax results"
[eq_img]: ./ex_imgs/equalized_img.png "Equalized img"
[gray]: ./ex_imgs/gray_img.png "Gray"
[rgb]: ./ex_imgs/rgb_img.png "RGB"
[good_train_res]: ./ex_imgs/improved_training_results.PNG "Improved training"
[bad_train_res]: ./ex_imgs/initial_training_results.PNG "Original training"
[signs]: ./ex_imgs/sign_examples.PNG "Signs"
[sign1]: ./ex_imgs/sign_2.png "Traffic Sign 1"
[sign2]: ./ex_imgs/sign_12.png "Traffic Sign 2"
[sign3]: ./ex_imgs/sign_13.png "Traffic Sign 3"
[sign4]: ./ex_imgs/sign_15.png "Traffic Sign 4"
[sign5]: ./ex_imgs/sign_36.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickle library to load the provided dataset into the jupyter notebook. Once the dataset was loaded the size of the various sets was found using the python len() function which returns the length of list or array. The size of the training images was found using the shape member variable of the image objects stored in the arrays. The number of classes was found by making a copy of the training label array and then converting it to a set. Sets in python can only have unique entries with no repeats so the conversion to a set kept only 1 label from each of the 43 classes.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32x32x2)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The image below shows my visualization of the dataset. I looped through the 43 different classes, found the first reperesentative of that class in the training set, and then used matplotlib to plot the image. The class number as well as the number of representatives from that class within the training set are also shown above each of the sign pictures. Just based on the counts of each sign it is easy to see that the dataset is poorly balanced in terms of sign example distribution. Some classes like the class 0 sign only has 180 images while the class 1 sign has 1980 images in the training set.

![alt text][signs]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Each image in the training, test, and validation set were run through the preprocessing pipeline using python's list comprehension functionality. The images are first converted to grayscale using OpenCV's cvtColor function. An example of an image before and after grayscale conversion can be seen below.

![alt text][rgb] ![alt text][gray]

The conversion to grayscale was performed mostly so that the image could undergo histogram equalization to increase the contrast of the image. As shown in the data visualization image there are a fairly large number of images that were taken in a very dark environment and thus have very poor contrast. OpenCV's equalizeHist function was applied to all of the images to increase the contrast of the image. For images that already had good contrast this didn't change much but for dark images this greatly improved the contrast. An example of the output from this step can be seen below.

![alt text][eq_img]

The final preprocessing step was to standardize the data by giving it a 0 mean and then dividing all of the values by the standard deviation. Standardizing the data was a common recommendation on a number of data preparation articles that I had read earlier and showed great results during the training of the network. Code for standardizing the validation set can be seen below. The np.finfo('float32').eps is added to the standard deviation before division to prevent a divide by 0 error. I don't know if this is possible but jupyter notebook kept showing warnings about a potential divide by 0 error. This adds the smallest possible value for a 32 bit float so the addition to the standard deviation is negligible. The network was training with and without the addition and there was no percievable difference.

```python
X_valid -= np.mean(X_valid, axis=0)
X_valid /= (np.std(X_valid, axis=0) + np.finfo('float32').eps)
```

Adding additional images to the training set was considered and partially implemented but after further testing improving the preprocessing steps was more than sufficient to improve network performance. The ImageDataGenerator class from keras was used to produce augmented data that had various transformations applied to the original training images. This would be an excellent future addition to the network as it would likely increase performance even further. When this was initially attempted it significantly increased training time so it was not viable with the 30 epoch setup that I found to work best with my network. With a more powerful computer this would likely be more viable.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is a modified version of the LeNet network from the previous lesson. I initially used it just to see how well it would perform but after seein how the results were I figure building on top of it would be the right way to go instead of trying to reinvent the wheel. The model can be seen below with desciptions of each layer as comments in the code. The main changes from the original LeNet are dimensions, convolution kernel size, padding type, and a dropout layer after the first fully connected layer. Layer dimension were an area that needed major tuning as I learned what works best over time. Originally the network was significantly wider in the fully connected layers because I thought a gradual decrease in fully connected layer width would be idea. This involved using more fully connected layer which eventually proved to be unnecesary as they didn't seem to improve performance at all. I also tried to add additional convolutional layers with the idea that they would be useful in learning more features complex features to differentiate between classes in the complex dataset. After much testing I found that the 2 convolution layers were sufficient. Dropout layer placement was also tested in depth. I tried having multiple dropout layers in both the convolution layers and the fully connected layers but this didn't seem to increase performance at all and even hurt it a number of cases.

```python
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Convolutional Layer 1: Input = 32x32x3
    # Weights
    conv1_w = tf.Variable(tf.truncated_normal(shape=(3,3,3,64), mean=mu, stddev=sigma))
    # Bias for each of the depth layers
    conv1_b = tf.Variable(tf.zeros(64))
    # The actual convolution layer
    # Stride of 1 in all dimenstions, with zero padding
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    
    # ReLU Activation.
    conv1 = tf.nn.relu(conv1)

    # Max Pooling.
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # Convolutional Layer 2
    conv2_w = tf.Variable(tf.truncated_normal(shape=(3,3,64,128), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(128))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1,1,1,1], padding='SAME') + conv2_b
    
    # ReLU Activation.
    conv2 = tf.nn.relu(conv2)

    # Max Pooling
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    

    # Flatten the output of the convolution layers and connect it to the first fully connected layer
    fc0 = flatten(conv2)
    
    # Fully Connected 1. 
    fc1_w = tf.Variable(tf.truncated_normal(shape=(fc0.shape[1].value, 128), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(128))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b
    
    # ReLU Activation.
    fc1 = tf.nn.relu(fc1)
    
    # Dropout layer
    fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

    # Fully Connected 2. 
    fc2_w = tf.Variable(tf.truncated_normal(shape=(128, 64), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(64))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    
    # ReLU Activation.
    fc2 = tf.nn.relu(fc2)
    
    # Fully Connected 3
    fc3_w = tf.Variable(tf.truncated_normal(shape=(64, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    
    return logits
```


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The training method from the LeNet lab was used again as there was no real reason to change it based on the performance achieved. The Adam optimizer was used again and provided excellent performance. The epochs, batch size, learning rate, and dropout layer keep_rate were the primary hyperparameters that were tuned. The number of epochs had a huge influence on the performance of the training process. I slowly increased this number from 10 to 30 try to find the ideal number before arriving finally arriving at 30. Anything past that showed no significant increase in performance and could have possibly led to overfitting. During certain session less epochs would likely work but 30 was a safe compromise. Learning rate and keep rate only needed light tuning to achieve good results but these could still be improved further. The learning rate was incredibly fickle and would destroy the entire training process if set impoperly.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00
* validation set accuracy of .971 
* test set accuracy of .948

If a well known architecture was chosen:
* What architecture was chosen?

The LeNet architecture was chosen and then tweaked to fit the needs of the application.

* Why did you believe it would be relevant to the traffic sign application?

Convolutional neural networks are excellent for classification problems and LeNet had already proven itself to be highly effective in this area with number classification. I originally beleived that I'd need to add more convolutional layers to the network due to the increased complexity of having to distinguish between 43 classes instead of 10 but this proved to be not necessary with proper preprocessing of the dataset. Even without extensive prerocessing and tuning the network performed well. The results of the network with only small adjustmens and improvments can be seen below.

![alt text][bad_train_res]

After proper tuning, preprocessing, and network tweaking the results became significantly better. The training results from the final network can be seen in the image below. I also added the functionality to save checkpoints for the model using tensorflow's saver object. This allows a specific well performing epoch to be loaded and used again in the future. In this case the model perfomed best on epoch 23 so the model from that checkpoint would loaded and used in the future.
 
 ![alt text][good_train_res]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][sign1] ![alt text][sign2] ![alt text][sign3] 
![alt text][sign4] ![alt text][sign5]


All of them were randomly selected from a large dataset but none of them are particularly dificult. The first image is fairly blurry and the bottom portion of the 5 is partially missing. The 4th image is both small and fairly dark so the network might have a hard time getting good features from it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

The network was able to classify all 5 of the signs correctly after they were preprocessed. The preprocessing for this images involved an extra step of using openCV's resize function to resize the images to 32x32x3. All 5 of the images were different sizes between 27 and 60 pixels in width and height so they needed to be resized to be fed into the network. While the 100% accuracy may seem impressive the result is from an incredibly small dataset of fairly easy to classify signs. If the signs were dificult or more were tested I'm sure the accuracy of the network would go down to a value comparable to the test set results shown above.

![alt text][results]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cells underneath the "Predict the Sign Type for Each Image" text in the jupyter notebook. The code first loads the desired checkpoint of the trained network and then uses it to classify the 5 preprocessed images. The tensorflow softmax function is used to turn the output activations of the network into probabilities that represent the certainty of the network in its classifications. The top 5 softmax probabilites for the 5 signs can be seen below. The network is extremely confident in all of its predictions which is likely because the images are fairly easy to classify. If harder images were used or just different classes of then the certainties would almost definitely be lower.

![alt text][softmax_results]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


# Acknowledgements
I used the tensorflow, opencv, and matplotlib documentation extensively when trying to debug issues or figure out how to use certain functions from the libraries. I of course also heavily used the materials from this course including the code all throughout the project. I also used the following sites as references for the sign dataset, data preparation, and LeNet.

* https://benchmark.ini.rub.de/gtsrb_news.html
* https://towardsai.net/p/data-science/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff
* http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf


