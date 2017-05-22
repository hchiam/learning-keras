# create sequential model
from keras.models import Sequential
model = Sequential()

# add layers to model
from keras.layers import Dense, Activation
model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# set up model learning settings
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# # or do even more setup
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

# train model (while iterating over trainind data in batches)
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)

# # or
# # train model (while manually giving batches)
# model.train_on_batch(x_batch, y_batch)

# evaluate performance
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# make predictions on new data
classes = model.predict(x_test, batch_size=128)
