# Image-Classifier Project

Second project in `Intro to Machine Learning with TensorFlow` nano degree on udacity, with a notebook goes through the training &amp; evaluation processes and a CLI Application to use for classifying flowers. this model is trained using TensorFlow oxford flowers dataset.

**NOTE**: It's preferred not to run the training cell if you don't have GPU

## Installation
- Install [Anaconda](https://www.continuum.io/downloads)

- This project uses TensoFlow2, to install it run the following command in your terminal
```bash
pip install tensorflow
```
- If you don't have GPU, consider using [Google colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
## Run
In a terminal or command window, navigate to the top-level project directory Image-Classifier/ and run one of the following commands:
```bash
ipython notebook Project_Image_Classifier_Project.ipynb
```
or 
```bash
jupyter notebook Project_Image_Classifier_Project.ipynb
```

## CLI Application
A Python script that runs from the command line / terminal.
**Basic Usage**
```bash
python predict.py /path/to/image saved_model
```
with saved_model file name = `tested_model_HDF5.h5`
**Options**
- --top_k: returns the top K most likely classes.
```bash
python predict.py /path/to/image saved_model --top_k KKK
```
-  --category_names: Path to a JSON file mapping labels to flower names.
```bash
python predict.py /path/to/image saved_model --category_names map.json
```
with json file = `label_map.json`
## Data
The data for this project is quite large, it's oxford flowers dataset with 102 different types of flowers. with 1020 examples for training and 1020 examples for validation and 6149 examples for testing.
![examples](Flowers.png)

