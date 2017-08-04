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
