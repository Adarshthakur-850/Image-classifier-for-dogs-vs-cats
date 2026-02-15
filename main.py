import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data_loader import load_data
from src.preprocessing import create_generators
from src.model_builder import build_cnn, build_transfer_model
from src.train import train_model
from src.visualization import plot_sample_images, plot_history
from src.evaluate import evaluate_model
from src.inference import predict_image

def main():
    print("Starting Dogs vs Cats Classifier Pipeline...")
    
    load_data()
    
    train_gen, val_gen, test_gen = create_generators()
    
    plot_sample_images(train_gen)
    
    print("Building and training CNN from scratch...")
    cnn_model = build_cnn()
    history = train_model(cnn_model, train_gen, val_gen, epochs=5)
    plot_history(history)
    
    evaluate_model(cnn_model, test_gen)
    
    print("Pipeline completed.")

if __name__ == "__main__":
    main()
