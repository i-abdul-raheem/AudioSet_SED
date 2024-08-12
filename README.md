# AudioSet Sound Event Detection v1.0

This project is designed to train a machine learning model and make predictions using Python scripts. The following instructions will guide you through setting up your environment, training the model, and making predictions.

## Setup

### 1. Install Dependencies

Before running any of the scripts, you need to install the required Python packages. Ensure you have Python installed, and then run:

```bash
pip install -r requirements.txt
```

This will install all necessary packages listed in the requirements.txt file.

### 2. Using the Makefile
The Makefile provided offers a convenient way to run the scripts for training and prediction.
Available Commands

`make train`: Runs the train.py script to train the model.

`make train-cached`: Runs the train.py script with the --cached option to use cached data for training, if available.

`make predict`: Runs the predict.py script to make predictions using the trained model.

### 3. Example Usage
To train the model, you can run:
```bash
make train
```
If you want to use cached data (if applicable), run:
```bash
make train-cached
```
After training the model, you can make predictions by running:
```bash
make predict
```

## Additional Information

Ensure your train.py and predict.py scripts are properly configured and located in the same directory as the Makefile.
Modify the train.py and predict.py scripts as needed to suit your specific use case.
