import os
import tensorflow as tf
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

models = tf.keras.models
image = tf.keras.preprocessing.image
load_model = tf.keras.models.load_model
IMG_SIZE = 150

# model files directory
model_path = 'put here model directory folder'
model_dir = os.path.join(model_path,'model.json')
weights_dir = os.path.join(model_path,'weights.h5')

# load json and create model
json_file = open(model_dir, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_dir)
loaded_model.layers[0].input_shape #(None, 160, 160, 3)
print("Loaded model from disk")


# load batch of images
batch_holder = np.zeros((30, IMG_SIZE, IMG_SIZE, 3))
img_dir = os.path.join(model_path,'batch/')
for i,img in enumerate(os.listdir(img_dir)):
    img = image.load_img(os.path.join(img_dir,img), target_size=(IMG_SIZE,IMG_SIZE))
    batch_holder[i, :] = img

labels = ["bus","camioneta","carro","motocarro","motocicleta","taxi"]

result=loaded_model.predict_classes(batch_holder)

# print and show the results
print(result.shape)
fig = plt.figure(figsize=(20, 20)) 
for i,img in enumerate(batch_holder):
    fig.add_subplot(5,6, i+1).axis('off')
    plt.title(labels[result[i]])
    plt.imshow(img/256.)
plt.show()
plot_img_dir = os.path.join(model_path,'multiclass6.png')
fig.savefig(plot_img_dir)

print(tf.test.is_gpu_available())