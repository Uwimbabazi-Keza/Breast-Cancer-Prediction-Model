from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l1, l2
from keras.optimizers import Adam

def train_model(x_train, y_train):
    model = Sequential()
    model.add(Dense(16, input_dim=30, activation='relu', kernel_regularizer=l1(0.01)))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, verbose=1, epochs=100, batch_size=64, validation_split=0.2)
    
    return model, history
