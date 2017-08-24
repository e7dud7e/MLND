README

#System setup
- Use Python 2.7
- Install Tensorflow 1.0 or newer
- Install python packages: matplotlib, sci-kit learn, sci-kit image, urllib3, tqdm

# Getting data
Go to the Kaggle competition site's [data page](https://www.kaggle.com/c/passenger-screening-algorithm-challenge/data) and click the blue "download" button to get the .aps data that I use for this project.

Copy the .aps data into the same parent folder of the jupyter notebook, in the directory: ./data/aps/

Copy the stage1_labels.csv and stage1_sample_submission.csv files into ./data/labels/

#Setup VGG
Download Udacity's git repository
```
git clone https://github.com/udacity/deep-learning.git
```
Then copy the folder ./transfer-learning/tensorflow_vgg to the same parent directory of the jupyter notebook.

Use the mlnd_capstone.ipynb code to download the vgg16.npy file

#Run the code
Open the mlnd_capstone.ipynb file to run the code.
Run most of the code, except where it specifies to run either one or the other. 
- For the "split into training, validation and test", Choose either option 1 or option 2.
- After the section "Train the Model", followed by the function plot_losses, skip to the cell that says "Trial 8" to plot the losses if you used Option 1 when splitting the training and test data.  Otherwise run the cell that says "Trial 8b" if you used option 2.
- After, run "predictions on test sample" until you get to "Visualization to summarise project."
- The "Visualization to summarize project" section should use option 2 so that there are 300 test samples.

