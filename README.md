# lenet-5-minst
Train a LeNet-5 neural network with Keras for minst classification

## Training
To train the network:
`python train.py`

Monitoring with tensorboard:
`tensorboard --logdir=logs/`

and open http://localhost:6006

## Classification
To evaluate for the 3 images:
`python classify.py`

# Dataset

- Udacity simulator traffic light images (173 images)
- Udactiy parking lot loop traffic light images (48 images)
- Udactiy parking lot traffic light images (118 images)
- Bosch traffic light train images (10698 images)
- Bosch traffic light test images (13486 images)
- Bosch traffic light additional images (320 images)
