from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt

def train_model(model, train_gen, val_gen, epochs=10):
    if not os.path.exists("models"):
        os.makedirs("models")
        
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("models/best_model.h5", save_best_only=True)
    ]
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )
    return history
