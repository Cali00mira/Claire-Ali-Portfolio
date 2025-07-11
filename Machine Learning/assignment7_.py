from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model, optimizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.initializers import RandomNormal, Constant
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.transform import resize
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

(X_train,y_train), (X_test,y_test) = cifar100.load_data(label_mode='fine')

# reshaping
X_train = X_train.reshape(-1, 32,32, 3)
X_test = X_test.reshape(-1, 32,32, 3)

# scaling
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255

# one-hot encoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test) 

# build the CNN architecture
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32, 32, 3),padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.2))

model.add(Dense(100))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])

# train CNN
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
model.fit(X_train, y_train_one_hot, batch_size=128,epochs=50,shuffle=True, callbacks=[early_stopping])

# evaluate the trained CNN on the test data
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)
print('Test loss', test_loss)
print('Test accuracy', test_acc)

# plotting random images with predicted labels
class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

fig = plt.figure(figsize=(12,12))

for i in range(5):
    idx = random.randint(0, len(X_test)-1)
    img = X_test[idx]
   
    img_pred = np.expand_dims(img, axis=0)
    pred = model.predict(img_pred)
    label1 = np.argmax(pred)
    label = class_names[int(label1)]

    img_disp = resize(img, (100,100), anti_aliasing=True)
 
    ax = fig.add_subplot(1, 6, i+1)
    ax.imshow(np.squeeze(img_disp))
    ax.set_title(f'Prediction: {label}')
    ax.axis('off')

plt.tight_layout() 
plt.show()