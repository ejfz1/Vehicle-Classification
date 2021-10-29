import os
import tensorflow as tf
VGG16 = tf.keras.applications.VGG16

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
print(conv_base.summary())

#	1. Directory input from the dataset
base_dir = 'Put here the base directory of the dataset'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

#   2. Adding a densely connected clasiffier on top of the convolutional base
models = tf.keras.models
layers = tf.keras.layers
tf_json = tf.keras.models.model_from_json

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(6,activation='softmax'))
print(model.summary())

print('This is the number of trainables weights before freezing the conv base:',len(model.trainable_weights))
conv_base.trainable = False
print('This is the number of trainables weights after freezing the conv base:',len(model.trainable_weights))

#   3. Training the model end to end with frozen convolutional base
image = tf.keras.preprocessing.image
optimizers = tf.keras.optimizers

train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='categorical')

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])
    
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

#   4. Plotting the results
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#   5. Freezing all layers up to a specific one
conv_base.trainable = True
set_trainable = False
for lay in conv_base.layers:
    if lay.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        lay.trainable = True
    else:
        lay.trainable = False

#   6. Fine-tuning the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-5),
    metrics=['acc'])
    
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

#   7. Saving the model
model_dir = os.path.join(base_dir,'vehicle6_model.json')
weights_dir = os.path.join(base_dir,'vehicle6_weights.h5')
model_json = model.to_json()
with open(model_dir,"w") as json_file:
    json_file.write(model_json)
model.save_weights(weights_dir)
print("Model saved")
#model.save(model_dir,include_optimizer=False)

#   8. Smoothing the plots
def smooth_curve(points,factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs,smooth_curve(acc),'bo',label='Smoothed training acc')
plt.plot(epochs,smooth_curve(val_acc),'b',label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,smooth_curve(loss),'bo',label='Smoothed training loss')
plt.plot(epochs,smooth_curve(val_loss),'b',label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#   9. Test model
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='categorical'
)

test_loss, test_acc = model.evaluate_generator(test_generator,steps=50)
print('Test acc:',test_acc)