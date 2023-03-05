#Import required libraries and packages.
import numpy as np
import cv2

from tensorflow.keras import applications

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

#All radiographs, stored as DICOM files, were converted into numpyz files composed of x_train and y_train.
#x_train corresponds to pre-processed radiographic images of size 224x224x3, transformed using the function img_transform.

img_width, img_height = 224, 224
def img_transform(image_dicom):
    image = image_dicom.pixel_array.astype(float)
    image_scaled = (np.maximum(image, 0) / image.max())

    if (image_dicom.PhotometricInterpretation == "MONOCHROME1"):
        image_scaled = 1 - image_scaled

    # Resize the image to 224x224 and normalize the pixel values.
    image_resize = cv2.resize(image_scaled, (img_width, img_height))
    image_resize = (image_resize - np.mean(np.array(image_resize))) / np.std(np.array(image_resize))

    # Apply standard normalization for the 3 color channels of the image, as used for training on the ImageNet dataset
    im_x = np.uint8((image_resize * 0.229 + 0.485) * 255.0)
    im_y = np.uint8((image_resize * 0.224 + 0.456) * 255.0)
    im_z = np.uint8((image_resize * 0.225 + 0.406) * 255.0)
    image_final = np.stack((im_x, im_y, im_z), axis=2)
    image_final = np.stack(image_final)[...,None]
    return image_final

#The output layer of model consists of five neurons, corresponding to the conditional probabilities of survival for five-time intervals with a spacing of a year (365 days).
breaks = np.arange(0, 1826.,365.)
n_intervals = len(breaks)-1

#y_train is an array that represents the ground truth for survival analysis, and it was generated using survival time and censor information.
y_train = make_surv_array(y_sur_train, censor_train, breaks)
y_val = make_surv_array(y_sur_val, censor_val, breaks)

#We used a pre-trained VGG19 model that was trained on the ImageNet dataset.
model_VGG19 = applications.VGG19(include_top=False,
                                 weights='imagenet',
                                 input_shape=(img_width, img_height, 3))

x = model_VGG19.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(n_intervals, 
                    input_dim = 1, 
                    kernel_initializer='zeros', 
                    bias_initializer='zeros', 
                    activation = "sigmoid")(x)

model_final = Model(inputs = model_VGG19.input,
                    outputs = predictions)

# Train the model using the Adam optimizer and a custom loss function and incorporated non-proportional hazards (Nnet-survival model; http://github.com/MGensheimer/nnet-survival)

from tensorflow.keras.optimizers import Adam

batch_size = 16
epochs = 100

early_stopping = EarlyStopping(monitor='val_loss', 
                               min_delta=0, patience=3,
                               verbose=1, mode='auto', restore_best_weights=True) 

model_final.compile(loss = surv_likelihood(n_intervals), 
                    optimizer = Adam(learning_rate=0.00001, beta_1 = 0.9))

history = model_final.fit(x_train, y_train, 
                          batch_size=batch_size, 
                          epochs=epochs,
                          validation_data = (x_val,y_val),
                          callbacks=[early_stopping])

model_json = model_final.to_json()
with open(path_model_json, mode='w') as json_file:
    json_file.write(model_json)
model_final.save_weights(path_model_weight)