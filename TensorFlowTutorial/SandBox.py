import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("First_DogvsCat_AI")

IMG_SIDE = 160

picture_path = r"E:\Animals_For_AI\Lenin.png"

class_names = ['cat', 'dog']

img = tf.keras.preprocessing.image.load_img(
    picture_path, target_size=(IMG_SIDE, IMG_SIDE)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
