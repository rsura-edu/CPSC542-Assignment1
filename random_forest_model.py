import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# function for loading and preprocess images
def load_images(folder_path, target_size=(240, 240)):
    images = []
    labels = []

    for label, celebrity_folder in enumerate(os.listdir(folder_path)):
        celebrity_path = os.path.join(folder_path, celebrity_folder)
        if '.DS_Store' in celebrity_path: # mac generated file i had issues with
                continue
        for filename in os.listdir(celebrity_path):
            image_path = os.path.join(celebrity_path, filename)
            if '.DS_Store' in image_path: # mac generated file i had issues with
                continue

            # Load and resize images
            img = Image.open(image_path)
            img = img.resize(target_size)
            img_array = np.array(img)

            images.append(img_array.flatten()) # Flatten image into 1D array
            labels.append((celebrity_path.split('/')[-1])) # celebrity name without whole path to folder

    return np.array(images), np.array(labels)

# Load and preprocess images
images, labels = load_images('celeb_src')

# 80/20 TTS
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=314)

# creating and fitting random forest
rf_classifier = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
rf_classifier.fit(X_train, y_train)


# model evaluation
y_pred_test = rf_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
y_pred_train = rf_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Train Accuracy: {train_accuracy:.2f}")


# plotting confusion matrix
disp = plot_confusion_matrix(rf_classifier, X_test, y_test, cmap=plt.cm.Blues, xticks_rotation='vertical')

plt.title("Random Forest Confusion Matrix")

plt.subplots_adjust(bottom=0.35)  # overflow labels on bottom of matrix
plt.savefig('rf_confusion_matrix.png') # save to image