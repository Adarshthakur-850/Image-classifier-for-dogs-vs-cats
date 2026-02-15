import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, test_gen):
    print("Evaluating model...")
    test_gen.reset()
    loss, acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    Y_pred = model.predict(test_gen)
    y_pred = (Y_pred > 0.5).astype(int)
    
    y_true = test_gen.classes
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()
