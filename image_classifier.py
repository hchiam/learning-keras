# example code to make use of the pre-trained image classification model ResNet50
# original source code: https://keras.io/applications/

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) # add 4th dimension since inputting 1 image but keras expects list of images
x = preprocess_input(x) # scale input to range used in ResNet50

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

# [('n02504458', 'African_elephant', 0.68460453), ('n01871265', 'tusker', 0.27114037), ('n02504013', 'Indian_elephant', 0.044251136)]
# identified African elephant image: https://upload.wikimedia.org/wikipedia/commons/6/6a/African_Forest_Elephant.jpg
