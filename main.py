import numpy as np
import keras
from keras import layers

def clean_data(data):
    (X_train, Y_train), (X_test, Y_test) = data

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = np.where(X_train >= 0.5, 1, 0).astype('int8')  # (60000 x 784) binary values
    X_test = np.where(X_test >= 0.5, 1, 0).astype('int8')  # (10000 x 784) binary values

    Y_train = keras.utils.to_categorical(Y_train, 10)  # (60000 x 10) 1-hot encoded
    Y_test = keras.utils.to_categorical(Y_test, 10)  # (10000 x 10) 1-hot encoded

    return X_train, Y_train, X_test, Y_test

def train_new_model(model_name):
    X_train, Y_train, X_test, Y_test = clean_data(keras.datasets.mnist.load_data())

    model = keras.Sequential(
        [
            keras.Input(shape=(784,)),
            layers.Dense(10),
            layers.Activation('relu'),
            layers.Dense(10),
            layers.Activation('softmax')
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_train, Y_train, batch_size=128, epochs=15, validation_split=0.1)
    model.save(model_name)

    loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)
    print("Test Loss", loss_and_metrics[0])
    print("Test Accuracy", loss_and_metrics[1])

def check_overflow(x, num_bits):
    if (x > (2 ** (num_bits-1) - 1)) or (x < -(2 ** (num_bits-1))):
        print('overflow detected')

def forward(model_name, iterations=10000):
    X_train, Y_train, X_test, Y_test = clean_data(keras.datasets.mnist.load_data())

    model = keras.saving.load_model(model_name)
    weights1 = model.layers[0].get_weights()[0]
    biases1 = model.layers[0].get_weights()[1]
    weights2 = model.layers[2].get_weights()[0]
    biases2 = model.layers[2].get_weights()[1]

    def to_fixed(float_value, bits_past_radix=10000000):
        a = float_value * bits_past_radix
        b = int(a)
        if a < 0:
            b = ~(abs(b)) + 1
        return b

    weights1 = np.vectorize(to_fixed)(weights1).astype('int8')
    biases1 = np.vectorize(to_fixed)(biases1).astype('int8')
    weights2 = np.vectorize(to_fixed)(weights2).astype('int8')
    biases2 = np.vectorize(to_fixed)(biases2).astype('int8')

    count = 0
    total = 0

    for X, Y in zip(X_test, Y_test):
        # HIDDEN LAYER

        output = [0] * 10
        for neuron in range(10):
            weights = weights1.T[neuron]

            weight = 0
            for index, pixel in enumerate(X):
                if pixel == 1:
                    weight += weights[index]
                    check_overflow(weight, 10000000)

            weight += biases1.T[neuron]
            check_overflow(weight, 10000000)

            output[neuron] = int(weight)

        hidden_out = np.array(output).astype('int8')

        # RELU

        hidden_out = np.maximum(0, hidden_out).astype('int8')

        # OUTPUT LAYER

        output = [0] * 10
        for neuron in range(10):
            weights = weights2.T[neuron]

            weight = 0
            for index, value in enumerate(hidden_out):
                weight += weights[index] * np.int16(value)
                check_overflow(weight, 10000000)

            weight += biases2.T[neuron]
            check_overflow(weight, 10000000)

            output[neuron] = int(weight)

        output_out = np.array(output).astype('int16')
        prediction = np.where(output_out == np.max(output_out), 1, 0).astype('int16')

        if np.array_equal(prediction, Y):
            count += 1
        total+=1

        if total >= iterations:
            break

        if total % 100 == 0:
            print(total)


    print(f'Accuracy: {count} / {total} = {count / total * 100}%')

def main():
    model_name = 'mnist_model.keras'
    # train_new_model(model_name)
    forward(model_name, 10000)

if __name__ == '__main__':
    main()
