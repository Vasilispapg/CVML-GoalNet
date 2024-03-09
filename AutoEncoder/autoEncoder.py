from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

def create_and_train_autoencoder(features, encoding_dim=1024, epochs=50, batch_size=32):
    input_dim = features.shape[1]

    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(input_dim, activation='relu')(input_layer)
    for dim in [input_dim//2, input_dim//4, input_dim//8, input_dim//16, input_dim//32, input_dim//64, encoding_dim]:
        encoded = Dense(dim, activation='relu')(encoded)

    # Decoder
    decoded = encoded
    for dim in reversed([ input_dim//64, input_dim//32, input_dim//16, input_dim//8, input_dim//4, input_dim//2, input_dim]):
        decoded = Dense(dim, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)  # Match the original input size

    # Autoencoder Model
    autoencoder = Model(input_layer, decoded)
    
    # Encoder Model
    encoder = Model(input_layer, encoded)

    # Compile and train
    optimizer = Adam(learning_rate=0.1)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    autoencoder.fit(features, features, epochs=epochs, batch_size=batch_size, shuffle=True)

    return encoder


def reduce_features_with_autoencoder(features, encoding_dim=1024, epochs=10, batch_size=256):
    """
    The `reduce_features_with_autoencoder` function takes in a set of features, creates and trains an
    autoencoder model to reduce the dimensionality of the features, and returns the reduced features.
    
    :param features: The input features that you want to reduce dimensionality for
    :param encoding_dim: The encoding_dim parameter determines the dimensionality of the encoded
    representation in the autoencoder. It specifies the number of neurons in the hidden layer of the
    encoder. The higher the value, the more information is preserved, but the higher the computational
    cost. It is a hyperparameter that needs to be tuned based, defaults to 256 (optional)
    :param epochs: The number of times the entire dataset is passed through the autoencoder during
    training, defaults to 50 (optional)
    :param batch_size: The batch size is the number of samples that will be propagated through the
    network at once. It is used to update the weights after each batch, defaults to 32 (optional)
    :return: The function `reduce_features_with_autoencoder` returns the reduced features obtained by
    encoding the input features using an autoencoder.
"""
    # Create and train autoencoder
    encoder = create_and_train_autoencoder(features, encoding_dim, epochs, batch_size)
    # Reduce dimensionality
    reduced_features = encoder.predict(features)
    return reduced_features
