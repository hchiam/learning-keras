blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

My Notes:

image classifiers with little data:
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
