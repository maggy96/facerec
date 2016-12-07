from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model 

def img_gen(path='data'):
    val_datagen = ImageDataGenerator(
            rescale=1./255)

    val_generate = val_datagen.flow_from_directory(
            path,
            target_size=(255, 255),
            batch_size=32,
            shuffle=False,
            class_mode='categorical')

    return val_generate


if __name__ == "__main__":
    val_gen = img_gen()

    model = load_model('face.h5')

    print(model.summary())

    score = model.evaluate_generator(
            val_gen,
            val_samples=5749)

    print("Validation correct (Accuracy) = {:.4f}".format(score[1]))
