# GAN_intro
An implementation of simple GANs
This code shows my attempt to implement some simple GANs to generate new images from MNIST and CIFAR datasets. 

The results for the MNIST dataset are quite good but the results from the CIFAR dataset are very bad. This is due to the fact that CIFAR images
are RGB images so they need more fine tuning of the network. I had some hardware constraints that prevented me from doing some good finetuning,
for example I couldn't use high batch size, which is a parameter that can drastically improve the results of a GAN.

