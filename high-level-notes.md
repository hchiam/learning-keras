# Use What When
keras = higher level coding that can use tensorflow or theano

tensorflow = more customization

keras = quick building and testing/experimenting

# Basic flow:

```py
# model
model = keras.models.Sequential()
# model.add(keras.layers.Dense())
model.add(Dense(32, input_dim=9)) # input layer
model.add(Dense(128))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

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
