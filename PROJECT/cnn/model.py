from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def build_model(params):
    lr, n_layers, filters, kernel, pool = params

    model = Sequential()

    for i in range(n_layers):
        if i == 0:
            model.add(Conv2D(filters, (kernel, kernel),
                             activation='relu',
                             input_shape=(28, 28, 1)))
        else:
            model.add(Conv2D(filters, (kernel, kernel), activation='relu'))

        model.add(MaxPooling2D(pool_size=(pool, pool)))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
