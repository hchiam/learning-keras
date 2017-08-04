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

# evaluate (new predictions)
model = keras.models.load_model('trained_model.h5')
predictions = model.predict(new_data)
```

You can also add ReLU activation function to each layer:

```py
model.add(Dense(32, input_dim=9, activation='relu'))
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

