# Load an h5 model and test it on the camera using cv2

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#test_img = cv2.imread("test3.png")

# Load the model
model = keras.models.load_model('WasteClassificationNeuralNetwork/WasteClassificationModel.h5')
DIR = "WasteClassificationNeuralNetwork/WasteImagesDataset"
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(DIR, validation_split=0.1, subset="training", seed=42, batch_size=128, smart_resize=True, image_size=(256, 256))

classes = train_dataset.class_names
print(classes)

# Load the camera
cap = cv2.VideoCapture(0)


# Loop through the camera frames

while True:
    # Read the frame
    _, frame = cap.read()
    #frame = test_img

    # Convert the frame to grayscale

    # Resize the frame to 224x224
    resized = cv2.resize(frame, (256, 256))

    img_array = tf.keras.preprocessing.image.img_to_array(resized)
    img_array = tf.expand_dims(img_array, 0) 

    # Predict the frame
    prediction = model.predict(img_array)

    # Get the prediction
    print("Prediction: ", classes[np.argmax(prediction)], f"{prediction[0][np.argmax(prediction)]*100}%")

    # Display the prediction
    cv2.putText(frame, classes[np.argmax(prediction)], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)


    # Wait for the user to press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
