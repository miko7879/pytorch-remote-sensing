# aircraft-detection
Aircraft detection using the 'Airbus Aircraft Detection' dataset and Faster-RCNN with ResNet-50 backbone.

Original dataset can be found here: https://www.kaggle.com/airbusgeo/airbus-aircrafts-sample-dataset

To train the nextwork, first configure the ML parameters in the src/config.py file. Then run 'python engine.py' in the src directory. This will run for the configured number of epochs. You can see intermediate plots of training and validation loss in the output directory. Once trained, run 'python inference.py' in the src directory. This will draw bounding box predictions for all images in the data/extras directory and will write the images with predictions to the output/inference directory.
