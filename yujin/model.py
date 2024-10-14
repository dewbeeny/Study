from keras import models, layers

def model(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    model.add(layers.Conv2D(16, (10, 10), strides=(1, 1), padding='valid', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=(2, 1)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model