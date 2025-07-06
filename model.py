from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def build_cnn(input_shape=(256,256,3), num_classes=3, dropout_rate=0.3):
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)
