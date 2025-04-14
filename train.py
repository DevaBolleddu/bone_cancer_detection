# train.py
from models import create_model
from tensorflow.keras.optimizers import Adam
from preprocess import augment_data
from sklearn.model_selection import train_test_split

def train_model(base_model_name, X_train, y_train, X_val, y_val, lr=1e-4, batch_size=32, epochs=20):
    model = create_model(base_model_name)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              batch_size=batch_size,
              epochs=epochs)
    return model
