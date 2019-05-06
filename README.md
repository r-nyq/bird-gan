# Bird GAN

## Dataset
Using [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

![](images/real_samples.png)
*Training data sample*

## Requierments
- Pytorch
- Visdom
- Numpy

## Training
Run `python3 src/train.py`  
Optional arguments
- generator_model
- discriminator_model
- epochs

## Create image
Run `pythin3 src/image_generator.py model "model path"`  
Optional arguments
- image_path

## Results
![](images/fake_sampes_001.png)
*Generated sample after 1 epoch.*

![](images/fake_sampes_050.png)
*Generated sample after 50 epochs.*

![](images/fake_sampes_200.png)
*Generated sample after 200 epochs.*

![](images/fake_birds.png)
*Generated sample after 500 epochs*