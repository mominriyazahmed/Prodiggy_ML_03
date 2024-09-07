# Cat and Dog Image Classification using Support Vector Machine (SVM)

## Overview
This project implements a Support Vector Machine (SVM) to classify images of cats and dogs from the popular Kaggle dataset. The primary objective is to demonstrate how SVMs, which are powerful for binary classification tasks, can be applied to image data to distinguish between images of cats and dogs.


## Dataset
The dataset used in this project is the Dogs vs. Cats dataset provided by Kaggle, containing 25,000 labeled images of cats and dogs. The dataset is divided into training and testing sets:

Training set: 20,000 images (10,000 cats and 10,000 dogs)
Testing set: 5,000 images (2,500 cats and 2,500 dogs)

## Project Workflow

## Data Preprocessing:

Resizing Images: All images are resized to a standard size (e.g., 64x64 pixels) to ensure uniformity.
Feature Extraction: Convert images into numerical feature vectors using pixel intensities or more advanced techniques like Histogram of Oriented Gradients (HOG).
Normalization: Scale the features to have zero mean and unit variance to improve the performance of the SVM.

## Model Training:

An SVM model is trained on the preprocessed training data using the RBF (Radial Basis Function) kernel, which is effective for non-linear classification tasks.
Hyperparameter tuning is performed using grid search with cross-validation to find the best combination of parameters (e.g., C, gamma).

## Model Evaluation:

The trained SVM model is evaluated on the testing set using metrics such as accuracy, precision, recall, and F1-score.
A confusion matrix is plotted to visualize the model's performance.

## Results:

The SVM model achieves an accuracy of approximately XX% on the test set.
The results demonstrate the effectiveness of SVMs for binary image classification tasks when combined with appropriate preprocessing and feature extraction techniques.

## Requirements
Python 3.x
Libraries: numpy, pandas, scikit-learn, opencv-python, matplotlib, seaborn, scipy

## Future Improvements
Use more advanced feature extraction methods like Convolutional Neural Networks (CNNs) to enhance performance.
Implement data augmentation techniques to increase dataset diversity and improve model robustness.
Experiment with different kernel functions and SVM variations.

## Conclusion
This project showcases the application of Support Vector Machines (SVMs) in image classification tasks and demonstrates the importance of data preprocessing and feature extraction in achieving good results. While SVMs are powerful for binary classification, more advanced techniques like deep learning may yield even better performance for more complex image classification problems.

## Acknowledgments
Kaggle for providing the Dogs vs. Cats dataset.
## License
This project is licensed under the MIT License - see the LICENSE file for details.
