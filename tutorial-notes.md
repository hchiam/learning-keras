**Keras intro**

 * **btw**: Karpathy’s course notes: cs231n.github.io

 * **keras**
   * rising popularity
   * tensorflow (apparently) going to have official support for keras https://github.com/fchollet/keras/issues/5050
   * philosophy: user friendliness
   * build model, then compile

 * **sequential** modelling: https://keras.io/getting-started/sequential-model-guide/
   * a->b->c->d ; linear NN from input to output

 * **functional** modelling: https://keras.io/getting-started/functional-api-guide/
   * you can do branching NN models, with multiple inputs and outputs at different places along the branching paths

 * not sure which **activation function** to use? —> ReLU (or maybe tanh)
   * **ReLU** = max(0,x) but risk “killing” NN :( but good for sparse representation :)
   * **tanh** is better than sigmoid b/c less saturation around ends and 0 maps to 0

 * **convolve** = filter (think image filters, like edge detection) —> = “get features”

 * **pool** = “chunk sections of the input”
   * (why? to reduce parameters; some data is actually redundant; like humans can work with bad resolution)
   * **example**: 4x4 —(max pool 2x2)—> each quarter —> just one 2x2 output

 * **upsampling** = opposite of maxpooling

 * **dropout** = “turn off” certain nodes/neurons —> why? so not all learning specific & generalizability & mitigate overfitting
   * how? ~ varying “width” of a layer at each epoch

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

**image classifiers with little data:**

 * that website uses kaggle data
 * data augmentation = create transformations of data already have
    * (con: risk overfitting <— high correlation)
    * (note: instatiate generator, then actually use it)
 * but mitgate that overfitting risk <— dropout
 * 80% accuracy already
 * use pre-trained network (e.g.: VGG16) —> 90% accuracy
    * VGG16 has a bunch of pre-trained layers
    * add “bottleneck features” selection somewhere before bottom/classification layer to apply to your specific classifier
 * can do even better than 90% (94%) if you fine-tune top layers of pre-trained network
    * freeze layers up to last convolution block (assumed already good)
    * fine-tune the last layer closest to the final fully-connected classifier
    * recommended: set a slow learning rate (lr) to not wipe out previous learning
 * can do even even better with other suggested approaches

https://blog.keras.io/building-autoencoders-in-keras.html

**simplest autoencoder example**

 * autoencoder is kind of like PCA or kind of like zipping files, but is more noisy (kind of like human memory? chunking?)
 * deep autoencoder = layered autoencoder
   * from this link: NOTE TO SELF: TRY OUT TENSORBOARD CALLABLE FROM INTO KERAS
VAE = generative model = creates probably likely output, instead of acting like a strict input-output function
   * = input —(function)—> likey/probable output

**more links:**

https://cs231n.github.io/neural-networks-1/

https://cs231n.github.io/convolutional-networks/

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

https://blog.keras.io/building-autoencoders-in-keras.html

https://github.com/fchollet/keras-resources

https://github.com/fchollet/keras/tree/master/examples

https://www.youtube.com/watch?v=u4alGiomYP4

https://www.youtube.com/watch?v=fTUwdXUFfI8
