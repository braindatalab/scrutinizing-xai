from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


def create_model_llr(output_dim, activation, input_dim, regularizer,
                     loss='categorical_crossentropy', l=0.01):
    """Creates a keras model with given input and output dimesnions, activation function, regularizer, and loss

    Input:
    output_dim : number of output dimensions
    activation : activation function corresponding to number of output dimensions
    input_dim : number of input dimensions
    regularizer : [None, l1, l2] type of regularization
    loss : loss function
    l : regularization strength

    Output:
    model : compiled model
    """
    model = Sequential()
    if regularizer is None:
        model.add(Dense(output_dim=output_dim, activation=activation,
                        use_bias=False, input_dim=input_dim))
    else:
        model.add(
            Dense(output_dim=output_dim, activation=activation,
                  use_bias=False, kernel_regularizer=regularizer(l), input_dim=input_dim))
    optimizer = Adam(lr=0.1)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs, verbose=0):
    """Trains the created model using a given training and test set

    Input:
    model : untrained keras model
    X_train : training set
    y_train : training labels
    X_test : test set
    y_test : test labels
    epochs : number of epochs for training
    verbose : [0, 1, 2] display of the training progress

    Output:
    model : trained model to use for interpretation methods
    weights : weights of the trained model for visualization
    """
    my_callbacks = [EarlyStopping(patience=30)]
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),
                        verbose=verbose, batch_size=64, callbacks=my_callbacks)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    return model, acc, val_acc
