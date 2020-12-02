This is a solution to [comma.ai's programming challenge](https://github.com/commaai/speedchallenge).

The challenge consists of predicting the speed of a car from a dash cam video given a limited amount of training data (20400 frames).

### Approach

`data/train.mp4`           |dense optical flow on `data/train.mp4`
:-------------------------:|:-------------------------:
 ![noflow](noflow.gif) |  ![flow](flow.gif)

For each pair of frames, I determined the [dense optical flow](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html). A convolutional neural network then extracts features from the dense flow image and 2 layers neural network learns those features to predict the speed per frame.

Approx. 20\% of the data is randomly chosen for validation, the rest for training. The validation/training partitioning can be changed by switching the ```numpy random seed``` in ```train.py```, ```prepare_data()```.

Training data is infinitely generated and augmented by changing it's brightness factor. The frames have the dashboard of the car and the sky cut out. Frame dimensions are then halved and color values are normalized to [-1, 1] before entering the neural network.

### Results
I trained the model for a total of 15 epochs on 3 different partitionings of training/validation data. The minimum validation loss was ```8.61```. I used the ```adam``` optimizer with a learning rate of ```1e-4``` and a batch-size of ```64``` for ```validation_loss > 20``` and ```32``` afterwards.

### Improvements
I think having more time and compute power would have yielded a better model and hence better results. Some improvements:
- find more ways of augmenting the training data: flip the frames, apply perspective transformations, flip the optical flow hue.
- experiment with deeper CNN models for feature extraction, namely ResNets and DenseNets
- make use of pretrained CNN networks
- train the model on several training/validation partitioning schemes
- (outside the scope of this challenge) find more data: checkout the [comma2k19](https://github.com/commaai/comma2k19) dataset
