# DNNMovieRS
Deep Neural Network based Movie Recommendation System Homework.
The model is implemented and trained on GPU using Pytorch with cuda.


## Requirements
> `numpy`<br/>
> `pandas`<br/>
> `scipy`<br/>
> `torch` // installed with cuda gpu

Install pytroch using following command in conda:
> `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`

## Description
### Model class definition
In this project I am using pytorch optimized with cuda to enable faster parallel computation on
GPU. The model is trained on our lab server computer with Nvidia GPU and stored to a file. When
prediction is required, the model is loaded from the file and prediction is made.

The model class has one input layer with 60 nodes. I have used 30 factors for user and movies here.
Then I have 3 hidden layers with 50, 30 and 20 nodes each. And finally output layer has only one
node. Every node is activated with ReLu activation function, which showed promising results and is
fast to calculate.

### Training
For training purpose, I am splitting the training data into batches of 256 size each. The model is
trained with Adam optimization. For loss function I am using just RMSE loss function, which is
provided by library. There are total of 50 epoch which showed good results. After model is trained
the parameters are saved to a file in the same directory.

### Prediction
Prediction process is very straight forward. I am doing just forward propagation activating all the
functions and outputting the result for each user id and movie id.

## Conclusion
This was the first time I am implementing neural network in python, and thus, I found it very
challenging. Nevertheless, I think I have learned a lot out of this project that I can use in the future.
I have learned how optimization functions work in practice. They help to accelerate the
code. And I have also learned that parallel processing is much faster than regular cpu. I had also
learned how to install cuda and how to benefit from it.