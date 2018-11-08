# Project: Robotics Nanodegree Follow Me

## Introduction
This write-up explains the ideas behind the fourth project of the Udacity Robotics Nanodegree, titled "Follow Me".

In this project, I programmed a drone to identify a specific person, refered to as the "hero", in a cluttered and crowded environment.
This made use of a technique called Semantic Segmentation, which classifies individual pixels in an image into different categories.
The idea behind this project was to collect data from a camera mounted on the drone, and individual frames collected would undergo
semantic segmentation to identify the location of the hero, the background, and the other passers-by in the scene. The semantic
segmentation was performed using a Fully Convolutional Neural Network (FCN), which was coded in Keras. A FCN is useful for this 
particular task of semantic segmentation, because while FCNs can be used to extract visual information from an image just like 
convolutional networks which end in fully connected layers, they also preserve spatial information throughout 
the entire network. Additionally, FCNs work on images of any size, as opposed to CNNs which end in fully connected layers, 
which are constrained by the dimensions of the fully connected layers.

## Network architecture
The architecture of the FCN can be summarized as three blocks - an encoder block, a 1x1 convolution layer, and a decoder block.

### Encoder
The role of the encoder is to extract features from the image, just as in a standard CNN. The layers closer to the input learn
to extract simple features such as edges, lines and curves. The succeeding layers combine these features into successively more
complex features. The encoder block is comprised of separable convolution layers, followed by batch normalizing layers.  
**Separable convolutions** comprise of a convolution performed over each channel of the previous layer, followed by a
1x1 convolution that takes the previous channel's outputs and combines them into an output layer. The advantage of separable
convolutions over regular convolutions is a drastic reduction in the number of parameters in a separable convolutional layer.
This results in improved runtime speed and performance.  
**Batch normalization** is a technique used to improve neural network performance, by normalizing the data that is passed to
each successive layer in the neural network, such that it has close to zero mean and unit variance. This allows us to train
neural networks faster, and use higher learning rates.  
In my chosen architecture, I used two successive blocks of separable convolution + batch normalization. The first block had 32
filters and the second had 64 filters. I chose these filters because as a rule of thumb, powers of 2 are good numbers to work
with, when training a neural network on a GPU. I estimated that the relatively simple shapes used in the simulation could result
in features that were easy enough to learn with two encoding layers.

### 1x1 convolutional layer
This is a layer midway between the encoder and the decoder. It simply a convolutional filter with a kernel size of 1x1.
The role of the 1x1 convolutional layer is to flatten the data incoming from the encoder, while still preserving spatial information.
This is essential in the case of semantic segmentation, where the location of each pixel to be classified is significant.  
In my architecture, I used a 1x1 convolution with 128 filters. Once again, this was chosen as a power of 2 for GPU efficiency.

### Decoder
The decoder upscales the features extracted by the encoder, into an output that is the same size as the original input image.
The decoder in this project used layers of bilinear upsampling, as opposed to transposed convolutional layers. The advantage
of bilinear upsampling is better computational efficiency, since it has no learnable parameters. However, it does lose finer
details in the decoding process as compared to transposed convolutions.  
The decoder also makes use of **skip connections**, which are direct connections between an encoding layer and a
corresponding decoding layer. The advantage of skip connections is that the model uses information from the earlier layers,
which would otherwise have been lost or blurred out by the process of convolution. This allows the model to make more precise
segmentations.  
In my architecture, I used two decoder blocks. The first performed bilinear upsampling on the output of the previous 1x1
convolutional layer, and concatenated this output with the output of the first encoder block, to achieve a skip connection.
The second decoder block upsampled the output of this decoder block, and concatenated it with the original input.
One final layer in the model was a 2D convolutional layer that returned an image the same size as the original input.

## Hyperparameters
This project was completed by brute force optimization and fine tuning of hyperparameters. After several trial and error
iterations, I ended up with a learning rate of 0.0015, a batch size of 128 - chosen as a power of 2 for GPU efficiency, and
30 epochs. I left the steps per epoch, validation steps, and number of workers at their default (provided by Udacity) values.
I noticed after ~30 iterations that my validation loss was not decreasing further.

## Results
With the hyperparameters chosen above, I could achieve a final score of , with a final IoU of .

## Future Enhancements
I used exclusively the training data set provided by Udacity for this project. One possible way to improve performance would
be to augment this dataset with a lot more data, collected by flying the drone in the simulator. I would have to plan data
collection runs in such a way as to obtain several different viewing angles and distances of the hero by the drone, and in
several different regions of the map, in varying levels of crowdedness. Adding several well-planned image runs in this way
would be a promising method of improving performance. Another possible enhancement would be to use regularization techniques
such as dropout in the encoding layers, to prevent overfitting on the training set. Also using an adaptive learning rate instead
of a fixed learning rate would improve performance as well. There is of course still the possibility of fine-tuning the
hyper-parameters even more, possibly with some kind of evolutionary algorithm.
