# Prediction-Uncertainty-for-Neural-Networks

## Solving the Problem of Prediction Uncertainty in Deep Neural Networks

Algorithms found in self-driving cars are only made possible with the data collected from vehicles equipped with numerous cameras. These cameras collect images of the car’s journey, capturing telephone poles, intersections, traffic lights, and more. Much like a human, the algorithm that powers a self-driving car must be provided with examples so that it can understand what to avoid, where to drive, and how to navigate complex intersections. Images that are collected from vehicles are used to train a Deep Neural Network so that when you or I hop in our first self-driving car, an algorithm can recognize objects and images on the road that it has not seen before and adapt appropriately and safely.

There are myriad challenges with efficiently training Deep Neural Networks for self-driving cars, and this project aims to address one of those challenges. When analyzing example images, we would like a method to calculate how sure we are that a certain object or pattern is present in the image. Deep Neural Networks (DNNs) used in regression tasks have lacked a principled method for calculating prediction uncertainty. Recently, Amini et.al [1] have shown how to use a Normalized Inverse Gamma function (NIG) to provide such a measure.

In this project, I will be exploring Amini et. al.’s method by first creating my own dataset and implementing a Neural Network in Python through the following steps:

1. Use the Pillow package in Python to generate a test and train dataset of varying artificially generated images of a straight road. These images will be constructed simply from shapes using Python.
2. Implement a Convolutional Neural Network in PyTorch that can predict the center of the lane where a car should be driving (the input will be the images, and the output will be two values which will represent the location of the center of the lane).
3. Add noise to the images and measure the effect on the training and validation loss given different noise levels
4. Understand Amini et. al.’s method of using a Normalized Inverse Gamma function to determine epistemic uncertainty

Future Work: Finish the process of converting Amini et. al.’s Github repository to PyTorch, and implement his method for this problem. Also, use the KITTI Vision dataset [2] which consists of real images, and apply the findings to this real-world data.

A big challenge in gathering training data for self-driving car neural networks is dealing with the large quantities of data that are collected from vehicles. This project has application for real problems that exist in the self-driving space today – developing a benchmark to determine when to save a piece of training data can reduce costs in both the storage of data, and computation time when training new models. And of course, this project also has the potential to make self-driving vehicles safer by allowing the driver to know when the car is not confident about its predictions.

I am reporting to and collaborating with Dr. Larry Jackel, President of North-C Technologies.

[1] [Prediction Uncertainty Paper](https://papers.nips.cc/paper/2020/file/aab085461de182608ee9f607f3f7d18f-Paper.pdf)
[2] [Kitti Dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php)

Reference to the Lenet5 neural network, and much of the code that was used for the train, validation, and evaluation of the model:
[Implementing lenet5](https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320)

## Python Scripts

* main.py: this is the main file that implements all of the other scripts in the repository
* Dataset.py: The Dataset class extends torch.utils.data.Dataset, and implements custom len() and getitem() methods for use with dataloaders in the main file
* RoadImage.py: Every training and validation image is represented by a RoadImage object which stored the actual image object and information about how that image was created
* roadNet.py: The PyTorch code for the actual neural network
* lenet5.py: PyTorch code for the Lenet5 Neural Network
* loader.py: The loader class stores all of the image data, and communicates with other classes to read and write data to the disk, and implement the dataloaders. It also contains a function to add noise to the images.