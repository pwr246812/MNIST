# Image classification

## Introduction

The main idea of the project is to implement a model, which will allow to classify clothes thumbnails. Its realization is divided into two parts. In first, implemented model was "written from a scratch" basing on k-nearest neighbors algorithm. In second part, libraries such as TensorFlow were used to create the model. 

### Part I
#### Methods

As it was said in the Introduction, implemented in this part model is based on an idea of k-nearest neighbors. It was chosen mainly because of its implementation simplicity. For those who, don't know this algorithm, here is an explanation. This is an example of non-parametric method. In fact prediction is made here on the basis of training set. Both feature extraction and learning algorithm are unnecessary here. New object is being compared to all of objects in training set. In comparison process we use various distance metrics. In this project "Euclidean distance" is used. Classes are being sorted from lowest distance value to highest. And here comes "k value" from the name of method. It tells how many classes (already sorted) are taken into account. The most common class in this trimmed set is a prediction for new object. Formally, the class with the highest probability value, but in this case it means the same as most common. Mentioned few lines up k-value is a hyperparameter. It can dramatically change the results of a model. That is why, it is so important to choose it carefully. The best way to show it is the image shown below.

![](https://miro.medium.com/max/1000/0*3gRZvgOB3-L4T5ow.png)

#### Results

To check how good model is it has to be checked on some testing set. Our testing set contains 10k images. This is the reason why it takes a while to calculate the accuracy of prediction. We must remember that every of this 10k objects is being compared to 60k objects from training set. It means 600kk operations. In order to shorten calculation time you can estimate accuracy only on some part of testing data. I will explain how to do it below. If you do not want to waste your time, you have to believe me that accuracy of this model on the whole testing set is about 85.5%. The highest I've found score on benchmark site for this method is 86% with training time about 42 minutes. As we can see mine model is not much worse in terms of accuracy but it is much slower. It is probably caused by suboptimal data structures or calculation methods. 

#### Usage
	
In order to run the file it is necessary to download training and testing set into your computer. Then in 8th and 9th line in KNN_model.py file you have to paste path to folder where downloaded data was placed. You also need to place mnist_reader.py file in the directory you placed KNN_model.py file. Beside that few basic libraries such as Numpy, PIL, Pandas and MatplotLib must be installed on your computer. From the technical site that is all. 
Now, the functions usage. There are some functions that are important only for model, not especially for user. User would probably like to use this few:

* show_images(columns, rows) - allows user to see images from training set. It takes two params: columns and rows (integers) to get to know how many images user wants to see

* k_selector(examples, k_range) - gives a feedback about the best k-value (with the highest accuracy). It takes two arguments: examples (number of images from testing set, max. 10k) and k_range (number of consecutive values to check). Keep in mind that for 10k examples it has to make 600kk * k_range operations. It may take a long time. 

* test_model(k, examples) - gives a feedback about the accuracy of model on given part of test data. It takes two arguments: k (integer value of neighbors taken into account) and examples (same as before)

* calculate_label(new_X, k) - predicts class for new object. It takes two arguments: new_X (new image (array)) and k (same as before).

You do not have to remember this. At the end of KNN_model.py file I showed an example of usage for all of above functions. You just have to uncomment it. 

### Part II

#### Methods

In this part the k-nearest neighbors algorithm is no longer used. Here, the model is created using TensorFlow library. Thanks to that, we can make use of Convolutional Neural Network. It is a class of deep neural networks most commonly used to analyze visual imagery. In this case it is important to normalize datasets. Why? Two reasons. Sometimes, for example when we have two features, let's say: age of worker and his salary one of features has always higher value. It can negatively affect on prediction accuracy. If we divide each feature by its highest value we automatically solve this problem. Secondly, if the values are very high (for example Image), calculation of output takes a lot of computation time as well as memory. That being said, I divide trening and testing data by 255 which is the highest possible value for a pixel. Model is constructed with few layers. One is a pair of Conv2D and MaxPooling2D. Conv2D produces a map of filters that convolutional layer will learn. MaxPooling2D down-sample an input representation by reducing its dimensionality. At the end we have two Dense layers. It is basically an implementation of equation:
    output = activation(dot( input, kernel) + bias

Conv2D:

![](https://pyimagesearch.com/wp-content/uploads/2018/12/keras_conv2d_num_filters.png)

MaxPooling2D:

![](https://computersciencewiki.org/images/8/8a/MaxpoolSample2.png)

Layers take as params many values. One of them is activation. Keras has a few activation functions prepared for us. They give basically similar results and accuracy. The only one which is a little bit worse than others is exponential, I do not recommend you use it, but with others feel free. A set which is saved in file I found the best but maybe there is a better combination. Interesting is a difference in results between a 'softmax' and 'relu' activation in last layer. Try it if you want. 
Next step is compiling model. We are teaching it by minimizing loss, which in my model is calculated by Sparse Categorical Crossentropy. More important here is an optimizer. Again we have few to choose from. I have tested all of them and achieved best accuracy with one called 'adam'. In practice it is a stochastic gradient descent method. 
Finally we can fit model to training data. Of course we are using training dataset and epochs number which sets number of learning iterations. We have to be careful again. To high numer placed here can provide huge overfit and our model will be making more mistakes during predictions for new objects. After fitting model, we can test it of course on testing set. 

#### Results

Saved in file setup achieve from 90 to 91 percent accuracy and comparing to results from benchmark it is very good score. Training time equals about two minutes. 

#### Usage

From technical site the only difference to Part I is that there are two extra libraries needed: TensorFlow and Random. Importing data is the same as previously. In this file, there are no functions at all, and that is why the only thing that user has to do is to run this code. The model will be generated, compiled and tested. After that, as an example, randomly picked from testing set image will be used to make a prediction. Result will be visualized on a plot. 
