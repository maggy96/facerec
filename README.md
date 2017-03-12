# Facerec
Facial recognition experiments with a deep neural network (InceptionV3)

## Dependencies
* keras

## Data
Data can be downloaded [here](http://vis-www.cs.umass.edu/lfw/).
Create a *data* folder and extract the archive directly into there, so that the structure is as follows:

* data
  * Person 1
    * image.jpg
    * second_image.jpg
  * Person 2
    * ...

## Results
The network achieves ~97.5% accuracy (I know, I know, on the training set) after the training.
This could probably be improved more, but for a limited time effort, this makes a solid result. To really verify performance though, you should create a test set.<Paste>
