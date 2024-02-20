import numpy as np
import tensorflow as tf 
import os
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report, accuracy_score

# Constants/over-arching variables
train_dir = 'celeb_src_cnn/train'
val_dir = 'celeb_src_cnn/validation'
test_dir = 'celeb_src_cnn/test'
img_size = (224, 224)
batch_size = 20
num_classes = 17
epochs = 50

# Preprocessing
train_gen = ImageDataGenerator(rescale=1./255, shear_range=3.14/4, zoom_range=0.25, horizontal_flip=True).flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical')
test_gen = ImageDataGenerator(rescale=1./255, shear_range=3.14/4, zoom_range=0.25, horizontal_flip=True).flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
val_gen = ImageDataGenerator(rescale=1./255, shear_range=3.14/4, zoom_range=0.25, horizontal_flip=True).flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical')

# Model building/compiling/fitting
model: object # declared for out of scope referencing
model_file_name = 'model.h5'
try: # if model already built
    model = load_model(model_file_name)
except:
    vgg_model = VGG16(weights='imagenet', 
                    include_top=False, 
                    input_shape=(224, 224, 3))

    # prevent vgg weights from being updated
    for layer in vgg_model.layers:
        layer.trainable = False

    # create model with VGG
    model = Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(17, activation='softmax'))

    # Compile and fit model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        validation_steps=len(val_gen),
        callbacks=EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        epochs=epochs,
        steps_per_epoch=len(train_gen),
    )

    # Save the model
    model.save(model_file_name)
    
    # Train/Val accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.title('CNN Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train','validation'], loc='upper left')
    plt.savefig("cnn_model_accuracy.png")


train_loss, train_acc = model.evaluate(train_gen)
print(f'Train Accuracy: {train_acc}')
val_loss, val_acc = model.evaluate(val_gen)
print(f'Validation Accuracy: {val_acc}')
test_loss, test_acc = model.evaluate(test_gen)
print(f'Test Accuracy: {test_acc}')


# # confusion matrix
# y_test_pred = np.argmax(model.predict(test_gen), axis=1)
# y_test_actual = test_gen.classes

# cm = confusion_matrix(y_test_actual, y_test_pred, 
#                     #   cmap=plt.cm.Blues, 
#                     #   xticks_rotation='vertical',
#                       )

# plt.title("CNN Confusion Matrix")

# plt.savefig('cnn_confusion_matrix.png') # save to image

# Confusion matrix
y_test_pred = np.argmax(model.predict(test_gen), axis=1)
y_test_actual = test_gen.classes

cm = confusion_matrix(y_test_actual, y_test_pred)

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('CNN Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(os.listdir(test_dir)))
plt.xticks(tick_marks, os.listdir(test_dir), rotation=45)
plt.yticks(tick_marks, os.listdir(test_dir))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('cnn_confusion_matrix.png')  # save to image
plt.show()