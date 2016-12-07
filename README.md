# mnist-keras
My Kaggle.com entry for MNIST classifier.
This one got me 0.99657 accuracy.

## Structure
* 2 CNNs, with 16 5x5 pixel filters, ReLU activations.
* MaxPooling - 2x2 pool size with 2x2 strides
* Fully connected layer, 128 neurons, ReLU activations, 0.5 dropout for regularization
* final fully connected output layer, softmax activations.
* loss function is categorical cross entropy
* GD optimizer Adam

