# Machine Learning Engineer Nanodegree
## Capstone Project
Eddy Shyu
August 13, 2017

[//]: # (Image References)

[img_frequency_threats]: ./examples/frequency_of_threats_by_body_zone.jpg "frequency of threats"
[img_input_01]: ./examples/input_images_01_to_04.jpg "input 01"
[img_input_02]: ./examples/input_images_05_to_08.jpg "input 02"
[img_input_03]: ./examples/input_images_09_to_12.jpg "input 03"
[img_input_04]: ./examples/input_images_13_to_16.jpg "input 04"
[img_loss]: ./examples/loss.png "loss function"
[img_loss_chart_03]: ./examples/loss_chart_03.png "loss chart 3"
[img_loss_chart_04]: ./examples/loss_chart_04.png "loss chart 4"
[img_loss_chart_05]: ./examples/loss_chart_05.png "loss chart 5"
[img_loss_chart_06]: ./examples/loss_chart_06.png "loss chart 6"
[img_loss_chart_07]: ./examples/loss_chart_07.png "loss chart 7"
[img_loss_chart_08]: ./examples/loss_chart_08.png "loss chart 8"
[img_test_loss]: ./examples/mlnd_kaggle_test_loss.png "test loss kaggle"

## I. Definition
_(approx. 1-2 pages)_

### Project Overview
In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_


This project is a computer vision and object detection task to help identify dangerous objects during airport security screening.  The data is provided by the Transportation Security Administration of the United States, and includes 3-dimensional images of body scans, some which include objects that we want to detect automatically.  In addition to identifying these objects, the goal is to identify where on the body these objects are located.  The TSA defines 17 body regions on which objects can be detected.  Examples of these regions include the left forearm, the right waist, left ankle, etc.

The goal is to develop a aystem that views a body scan, then correctly predicts the probability that a threat is hidden on each of the 17 body regions.

This project is a Kaggle competition that is hosted by the Transportation Safety Administration.  The data is available through the Kaggle website in multiple formats.  Two of the data formats are 2D slices of the 3D image, at various angles, as if a camera is moving around the person and taking x-ray images at equally spaced intervals.  The smaller data set (.aps) has 16 angles.  The larger data set has 64 angles.  A third data format consists of 3 coordinates, and looks like several 2D slices stacked on top of each other.


### Problem Statement
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

My goal is to train a convolutional neural network that can view a new body scan and output a prediction between 0 and 1 for each of 17 body regions.  A prediction close to '1' should be output when there is a dangerous object present in the image, for that body region.  An output close to '0' should be ouput when that body region has no dangerous object present.

Since we have 3D data that is provided as multiple 2D slices, I will train a multi-view convolutional neural network.  The multi-view CNN reuses the same filter windows (reuses the weights of the convolutions) for all 2D slices.  Each convolutional layer has an output for each 2D slice, so for instance, if the data has 16 2D slices at different angles around the body scan, each convolutional layer will have 16 different outputs.

The multiple outputs are then combined with an aggregate function (in this case, a maximum), so that the 16 outputs are combined into one output with the same dimension as a single 2D slice.

I will use transfer learning to improve my prediction results and reduce training time.  I will use a pre-trained VGG network, and add a few trainable convolutional and fully connected layers to the end.


### Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_
I will measure the model performance using a the average log loss function.  The log loss function is minimized when the prediction is close to 1 and the actual value should be 1.  It is also minimized when the prediction is close to zero and the actual value is 0.  

The log loss function is also the metric that is used for the Kaggle competition.



## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_

There are 1147 training images.  Since these are provided by the TSA, the resolution and size of images are identical across samples, and the volunteers being scanned are of various sizes and genders, but are consistently placed in the center of the image.  The data format of the smallest size (.aps) has images in one channel, a height of 660, width of 512, and 16 angles.  

I display 16 sample images and the locations of the threats.  
![input example 01][img_input_01]
![input example 02][img_input_02]
![input example 03][img_input_03]
![input example 04][img_input_04]

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

I aggregate the label data to show the frequency that a dangerous object is hidden in each of the 17 body regions.  The frequency that a threat occurs on a particular body zone ranges from 7.8% to 11.6%, with an average of 9.6% across all training samples.
![frequency of threats][img_frequency_threats]


### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

#### Data Pre-processing
In order to fit the images into the VGG network, I size the images to be 224 by 224, with values that range from 0 to 1.  To avoid distorting the images, which are originally 660 high by 512 wide, I pad the image with zeros first, so that the shorter side (width) first equals the longer side (the height).  Once the image is square, I transform it from a 612 x 612 square to a 224 x 224 square.  

Also, since the VGG network expects the inputs to be 3 channel RGB images, I copy the single channel image into three channels before feeding the data into the network.

#### Neural Network
I am using a pre-trained VGG convolutional neural network to generate inputs into a smaller multi-view convolutional neural network.  I used the vgg16 network.  For vgg16, I took the output of conv5_3 (the 13th and last convolutional layer).

I am using transfer learning and the pre-trained VGG network because it provides more insight into lower level features, since it was trained on more images for a longer period of time.  

I added 3 convolutional layers to process the output of the pre-trained network.
convolution 1 has a 3x3 kernel, stride of 1, and depth 512.
convolution 2 has a 3x3 kernel, stride of 2, and depth of 1024
convolution 3 has a 3x3 kernel, stride of 1, and depth of 1024 

By condensing the height and width I attempt to aggregate lower level features into higher level features.  By increasing depth, I leave more room for different combinations of the lower level features.

Since the inputs are multiple 2D images for a single observation, I reuse the same weights for the trainable convolutional layers, for all angles. We still generate outputs for each of the angles, so if we have 16 angles, the convolutional layers also output 16 sets of activations.

Next, an aggregation step flattens the last convolutional layer and aggregates them by taking the maximum for each element position, across all 16 sets of tensors.  So it's in effect collapsing the set of 16 tensors from 16 angles into one tensor of the same shape.  From this point onward, the rest of the neural network is the same as a convolutional neural network that works with 2D images.

The next three layers are fully connected (dense) layers of decreasing size.  The first dense layer is 2048 units, the second is 512. The fully connected layers get progressively smaller, so that each successive layer aggregates the information from the previous layer.

The output layer has 17 units, to represent the 17 body region zones.  We pass the 17 logits through a sigmoid activation so that the output can be treated as a probability that ranges between 0 and 1.


### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_

The Kaggle leaderboard calculates an average log loss function on a test data set, and serves as a good source for a benchmark.  My goal is to reach a loss score of 0.30 or lower, as this is where a significant number of competitors rank.  The loss score for a completely naive 0.5 prediction is 0.69, so I definitely want to score below that.

Here is the loss function as described on the Kaggle competition's page.
![loss function][img_loss]


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

Since the data was generated by the TSA, there aren't any abnormalities or outliers that should be removed from the training or testing data.


### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_


#### Pre-trained network
For the VGG pre-trained network, I downloaded the version of tensorflow_vgg provided by Udacity, since the one directly from the original github had a bug when I ran it.
So I used 
```
git clone https://github.com/udacity/deep-learning.git
```
instead of using https://github.com/machrisaa/tensorflow-vgg.git


From there, I copied the deep-learning/transfer-learning/tensorflow_vgg folder.
Udacity has a convenient place on Amazon Web Services that stored the pre-trained weights for vgg16 (vgg16.npy).

In the get_codes_vgg function, I'm working with 3D data.  The data dimensions are: batch, angle, height, width, depth.
When I feed each 2D slice into the VGG network, I'm saving the outputs of the network into a list of size 16 (1 for each angle), so that each numpy array stored in the list has dimensions batch, height, width, depth.  

When I'm done collecting all the outputs of the pre-trained network, I convert the list of size 16 into one big numpy array, which has dimensions angle, batch, height, width, depth.  Since I want to keep the order of the dimensions the same as the input data, I use a transpose to re-order the dimensions into batch, angle, height, width, depth.  I save these "codes" to disk so that I can use them later.

I saved the training data codes that are output from the vgg16 network.  I did the same for the smaller test data set.

#### Trainable network

I built a small multi-view convolutional neural network that takes the outputs of the pre-trained network as its inputs.  I add some convolutional layers, and each layer includes a convolution, batch normalization, a leaky relu activation, then a dropout.

Since this is 3D data, I feed each of the 16 2D slices into the network in a loop.  I initialize weights for the first angle, but all subsequent angles reuse the weights.  I use a variable scope to reuse the weights.  I flatten the output of the final convolutional layer, and save these in a list of 16 elements, one for each angle.

Next, I use a view pooling layer that combines all 16 angles into a single tensor.  I use a reduce_max function, which lines up all 16 tensors, and for each position within those tensors, takes the maximum out of the 16, and saves that into a new tensor.  The output of this pooling layer has the same dimensions as one single tensor within the list of 16.

I pass this pooled layer into some fully connected layers.  The final output layer has 17 logits that represent the 17 body regions.  I pass these logits through a sigma activation so that each represents a probability between 0 and 1.

#### Training
I use a log loss function to measure the error between prediction and actual targets, as this is the same metric used for the kaggle competition.  I use an AdamOptimizer to perform back propagation, which uses a learning rate of 0.0001, and a beta1 of 0.5. 


### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_

I chose some hyper-parameters and mostly modified the number of epochs, and the design of the trainable network layers.  I used a low learning rate of 0.0001, beta1 of 0.5, batch size of 64, and dropout keep probability of 0.5

For the network design, I incrementally adjusted the number of layers, the number of filters (a.k.a. channels, or depth), and the stride size.

#### Trial 3
I first started with two convolutional layers and two dense layers.

 layer name | details |
convolution 1: | kernel 3x3, stride 2x2, valid padding, depth 512 |
convolution 2: | kernel 3x3, stride 2x2, valid padding, depth 1024 |
fully connected: | size 1024 |
fully connected: | size 256 |

| Layer Name    | Kernel/Filter Size | Stride | Padding | Depth / Channels |
| --- |:---:| :---:| :---:| :---:|
| convolution 1 | 3x3 | 2x2 | valid | 512 |
| convolution 2 | 3x3 | 2x2 | valid | 1024 |

| Layer Name | Size |
| --- |:---:|
| fully connected 1 | 1024 |
| fully connected 2 | 256 |
| output logits | 17 |

With 30 epochs, I get a validation loss of 0.3227, which follows the training loss closely.  
![loss chart 3][img_loss_chart_03]

#### Trial 4
Next, I change the stride of convolution 1 from 2x2 to 1x1.  The training and validation loss jump up and down more during training.  I trined for 20 epochs, it reach the best validation loss at epoch 16, at 0.3274.

| Layer Name    | Kernel/Filter Size | Stride | Padding | Depth / Channels |
| --- |:---:| :---:| :---:| :---:|
| convolution 1 | 3x3 | **1x1** | valid | 512 |
| convolution 2 | 3x3 | 2x2 | valid | 1024 |

| Layer Name | Size |
| --- |:---:|
| fully connected 1 | 1024 |
| fully connected 2 | 256 |
| output logits | 17 |

![loss chart 4][img_loss_chart_04]

#### Trial 5
I went back to using 2x2 strides for each convolutional layer, and added a third layer.  In order to keep the height and width of the output from shrinking as quickly, to allow for a third layer, I change padding from valid to same.  I also make the second convolution have a depth of 750 instead of 1024, and the third layer has a depth of 1024.
I trained for 20 epochs; it gets valid loss .3206 at epoch 17, and gets worse (increases after that).

| Layer Name    | Kernel/Filter Size | Stride | Padding | Depth / Channels |
| --- |:---:| :---:| :---:| :---:|
| convolution 1 | 3x3 | **2x2** | **same** | 512 |
| convolution 2 | 3x3 | 2x2 | **same** | **750** |
| **convolution 3** | **3x3** | **2x2** | **same** | **1024** |

| Layer Name | Size |
| --- |:---:|
| fully connected 1 | 1024 |
| fully connected 2 | 256 |
| output logits | 17 |

![loss chart 5][img_loss_chart_05]

#### Trial 6
I try to use 1 convolutional layer instead of 2 or 3.  I trained for 20 epochs, and it gets a validation loss of 0.3360.

| Layer Name    | Kernel/Filter Size | Stride | Padding | Depth / Channels |
| --- |:---:| :---:| :---:| :---:|
| convolution 1 | 3x3 | 2x2 | same | 512 |

| Layer Name | Size |
| --- |:---:|
| fully connected 1 | 1024 |
| fully connected 2 | 256 |
| output logits | 17 |

![loss chart 6][img_loss_chart_06]

#### Trial 7
I try increasing the size of the convolutional layer from 512 to 1024, to give more room for learning features.  I trained for 20 epochs, and got a validation loss of 0.3489.

| Layer Name    | Kernel/Filter Size | Stride | Padding | Depth / Channels |
| --- |:---:| :---:| :---:| :---:|
| convolution 1 | 3x3 | 2x2 | same | **1024** |

| Layer Name | Size |
| --- |:---:|
| fully connected 1 | 1024 |
| fully connected 2 | 256 |
| output logits | 17 |

![loss chart 7][img_loss_chart_07]

#### Trial 8
I try doublig the size of the fully connected dense layers.  I trained for 20 epochs, and the best validation loss at epoch 19 was 0.2898.

| Layer Name    | Kernel/Filter Size | Stride | Padding | Depth / Channels |
| --- |:---:| :---:| :---:| :---:|
| convolution 1 | 3x3 | 2x2 | same | 1024 |

| Layer Name | Size |
| --- |:---:|
| fully connected 1 | **2048** |
| fully connected 2 | **512** |
| output logits | 17 |

![loss chart 8][img_loss_chart_08]

## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

In order to evaluate the model, I use test data that I did not use during training.  This consisted of 100 samples from the original 1147 training samples, that I did not use in the training or in validation.  This resulted in a test loss of 0.2898, which is similar to the validation loss.

Another way I evaluated the model is by predicting on the test sample selected by Kaggle, which is a separate 100 samples that are not part of the 1147 training samples.  Since I don't have the labels for this test sample, I make the predictions and then upload them to Kaggle. Kaggle reports a loss score of 0.28421, which is below my goal of 0.29.

![kaggle test loss][img_test_loss]

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_

Kaggle provides a benchmark loss value for when all predictions are a 50% chance that every body part of every sample has a hidden threat; this average log loss is 0.69315.  I definitely wanted to get below this.

A benchmark that I set for myself was to reach a loss below 0.2900.  This is because there were many people on the Kaggle leaderboard scoring between 0.2900 and 0.30, ranging from a rank of 57 to 132 (as of August 22, 2017).

Since I reached a score of 0.28421, I reached my goal of reaching a loss lower than 0.2900.
This score puts me in the top 22% of the contestants as of August 22, 2017.

## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_



### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
