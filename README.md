# CPSC 542 - Assignment1

Link to [Github Repository](https://github.com/rsura-edu/CPSC542-Assignment1/tree/main).
Link to [external celebrity dataset](https://www.kaggle.com/datasets/bhaveshmittal/celebrity-face-recognition-dataset) (reuploaded to github for formatting sake)

## 1) Personal Info
- a. Full name: Rahul Sura
- b. Student ID: 2371308
- c. Chapman email: sura@chapman.edu
- d. Course number and section: CPSC 393 - 02
- e. Assignment or exercise number: HW 5

## 2) Source Files:
- random_forest_model.py
- cnn_training.py
- cnn_eval.py
- eda.py

## 3) A description of any known compile or runtime errors, code limitations, or deviations from the assignment specification (if applicable):
- Need to use the following versions to ensure no deprecated functions are used:
    - python version 3.8.10
    - sklearn version 0.24.2

## 4) A list of all references used to complete the assignment, including peers (if applicable):
- Discussed with Shree Murthy and Dylan Inafuku about preprocessing techniques and using VGG16
- https://scikit-learn.org/stable/whats_new/v0.24.html
- https://docs.python.org/3/library/os.html for file exploration using folders
- Old CPSC 393 code for VGG, data preprocessing, and layers
- Old CPSC 392 code for random forest

## 5) Instructions for running the assignment
- Random forest model:
    - `python3 random_forest_model.py`
- CNN model:
    - `python3 cnn_training.py`
    - `python3 cnn_eval.py`