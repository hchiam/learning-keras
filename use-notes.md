Based on: https://www.linkedin.com/learning/building-deep-learning-applications-with-keras-2-0

# Use What When:
keras = higher level coding that can use tensorflow or theano

tensorflow = more customization

keras = quick building and testing/experimenting

# Basic Flow:

```py
# model
model = keras.models.Sequential()
# model.add(keras.layers.Dense()) # or: from keras.layers import *
model.add(Dense(32, input_dim=9)) # input layer
model.add(Dense(128))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam') # mse = mean_squared_error

# train
mode.fit(training_data, expected_output)

# test
error_rate = model.evaluate(testing_data, expected_output)

# save
model.save('trained_model.h5')

# predict (on new data)
model = keras.models.load_model('trained_model.h5')
predictions = model.predict(new_data)
```

# Keras Comes With Pre-Trained Models Installed:

VGG, **ResNet50**, Inception-v3, and **Xception** can recognize 1000 objects.

You can also fine-tune/adapt them to recognize new objects too.

You'll need to reshape input data to match to the model's number of input neurons; "match the plug to the socket".

```py
from keras.preprocessing import image
img = image.load_img(image_file, target_size=(224, 224))
```

See more at https://github.com/hchiam/learning-keras/blob/master/image_classifier.py

# Special Layers:

## convolutional

= images/spatial data

```py
keras.layers.convolutional.Conv2D()
```

## recurrent

= memory for sequential data, like sentences, where previous data acts as "context"

```py
keras.layers.recurrent.LSTM()
```

# Shape Input Data:

Best practice: make all data in terms of range from 0 to 1. You can do that with sklearn: 

```py
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data1 = scaler.fit_transform(training_data1) # figure out and use transform (= x *... +...)
scaled_data2 = scaler.transform(training_data2) # apply the same transform
# rescale back to original units using: scaler.scale_[8] and scaler.min_[8]
```

# Activation Functions:

You can also add ReLU activation function to each layer:

```py
model.add(Dense(32, input_dim=9, activation='relu'))
```

And then a linear activation function for the final output:

```py
model.add(Dense(1, activation='linear'))
# or by default:
model.add(Dense(1))
```

# More Training Specs:

**Epochs:** Tell keras how many passes (epochs) to do. (Test to tune performance but also stop early to avoid overfitting.)

**Shuffle:** Shuffle the data.

**Verbose:** Show more details on the training print-outs.

```py
# train
model.fit(
  training_data, 
  expected_output, 
  epochs=50,
  shuffle=True,
  verbose=2
)

# test
error_rate = model.evaluate(testing_data, expected_output, verbose=0)
```

# Re-Shape Predictions:

To just get first value for first prediction:

```py
prediction = predictions[0][0]
```

Get back in original units:

```py
prediction = prediction - scaler.min_[8]
prediction = prediction / scaler.scale_[8]
```

# Reuse Model:

```py
model = keras.models.load_model('trained_model.h5')
predictions = model.predict(new_data)
prediction = predictions[0][0]
prediction = prediction - scaler.min_[8]
prediction = prediction / scaler.scale_[8]
```

# Add TensorBoard Logging

Create data files in format tensorboard can read.

```py
logger = keras.callbacks.TensorBoard(
  log_dir='logs', # you can use a sub-folder directory to get different runs to compare on TensorBoard
  write_graph=True,
  histogram_freq=5
)
```

Add to model training:

```py
model.fit(
  training_data, 
  expected_output, 
  epochs=50,
  shuffle=True,
  verbose=2,
  callbacks=[logger]
)
```

Name layers for easier reading:
```py
model.add(Dense(50, input_dim=9, activation='relu', name='layer_1'))
# etc.
```

In Terminal: `tensorboard --log_dir=<logs folder>` (make sure that `<logs folder>` is the parent folder of any sub-folders for different runs). Then go to the URL that prints out to see TensorBoard. Graphs tab = flow chart. Scalars tab = compare runs from different sub-folders.

# Use Trained Keras Model as TensorFlow Code in Google Cloud

Export as TensorFlow model:

```py
import tensorflow as tf
model_builder = tf.saved_model.builder.SavedModelBuilder('exported_model') # exported_model is folder to save in
inputs = {
  'input': tf.saved_model.utils.build_tensor_info(model.input) # just get input info from keras model
}
outputs = {
  'earnings': tf.saved_model.utils.build_tensor_info(model.output) # just get output info from keras model
}
# this "function definition" will be the same every time
signature_def = tf.saved_model.signature_def_utils.build_signature_def(
  inputs=inputs,
  outputs=outputs,
  method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
)
# save both structure and trained weights
model_builder.add_meta_graph_and_variables(
  K.get_session(), # reference to current keras session
  tags=[tf.saved_model.tag_constants.SERVING], # tag to know meant for serving users
  signature_def_map={ # pass in signature_def from above, and this is also same every time
    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature,
  }
)
model_builder.save()
```

In `exported_model` folder (as named above), there should be a `variables` folder and a `saved_model.pb` Google's protobuf format. Ready for uploading to the cloud!

## https://console.cloud.google.com

* Project
* 3 lines icon -> API Manager -> Library -> Google Cloud Machine Learning -> Machine Learning Engine API -> enable it!
* 3 lines icon -> Billing (to enable services)

## Google cloud SDK: https://cloud.google.com/sdk/downloads

* Ctrl+f: "Run the interactive installer to download and install the latest release"
* Interactive installer
* Install the SDK
* Activate the SDK in Terminal: `gcloud init`

## Upload and Use Cloud-Based Model

1. Upload bucket to cloud
2. Create model

* Terminal: navigate to model folder, and then create bucket: for example `gsutil mb -l us-central1 gs://keras-class-1000`
  * `mb` = make bucket
  * `1000` may be different for you
* Terminal: upload bucket: for example `gsutil cp -R exported_model/* gs://keras-class-1000/earnings_v1/`
  * `cp` = copy
  * `-R` = recursive, so sub-folders too
* Terminal: create new model: for example `gcloud ml-engine models create earnings --regions us-central1`
  * model will be called `earnings`
* Terminal: tell Google which files should be published as first version of model: for example `gcloud ml-engine versions create v1 --model=earnings --origin=gs://keras-class-1000/earnings_v1/`
  * `v1` = your defined version name
  * `--model` to specify model to create version under
  * (then wait)

Small data set to test on?

* Terminal: try it out: for example: `gcloud ml-engine predict --model=earnings --json-instances=sample_input_prescaled.json`
  * `--json-instances` to specify local input data file to try it on

Large data file?

* Upload file to cloud storage bucket.
* Use `gcloud` command to make prediction from that file.

Or:

* Use google cloud api client library for any supported programming language to call model from your program.

# Use Model from Google Cloud in Software Written in Any Programming Language

https://developers.google.com/api-client-library/

Need:
1. Permission/authorization, for security: credentials file.
2. Call API.

* https://console.cloud.google.com
* 3 lines icon -> API Manager -> Credentials -> Create Credentials -> Service Account Key
* "New Service Account" + create name + set role as "Project -> Viewer"
* Create -> should get a file -> rename to credentials.json -> save in folder
* In code: project id, model name, credentials file. Also have input data ready.

```py
from oauth2client.client import GoogleCredentials
import googleapiclient.discovery
credentials = GoogleCredentials.from_stream(CREDENTIALS_FILE)
service = googleapiclient.discovery.build('ml', 'v1', credentials=credentials)
name = # <directory to model>
response = service.projects().predict(
  name=name,
  body={'instances': inputs_for_prediction}
).execute()
if 'error' in response:
  raise RunTimeError(response['error'])
results = response['predictions']
print(results)
```
