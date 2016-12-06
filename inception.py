from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model 
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


def img_gen(path='data'):
    train_datagen = ImageDataGenerator(
        rescale=1./255)
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True)

    train_generate = train_datagen.flow_from_directory(
        path,
        target_size=(255, 255),
        batch_size=32,
        class_mode='categorical')
    
    return train_generate


if __name__ == "__main__":
    base_model = InceptionV3(include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    predictions = Dense(5749, activation='softmax')(x)

    model = Model(input=base_model.input, output=predictions)

    print("Compiling model...")
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    print("Compiled. Preparing Images...")
    train_gen = img_gen()
    print("Fitting Model")
    model.fit_generator(
        train_gen,
        samples_per_epoch=5749,
        nb_epoch=50)

    model.save('catdog.h5')
