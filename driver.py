# driver.py
from preprocess import preprocess_image, augment_data
from train import train_model
from evaluate import evaluate_model
from explainability import grad_cam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    # Simulated dataset loading
    images, labels = load_dataset()  # Replace with actual dataset loading
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

    # Train model
    model = train_model('ResNet50', X_train, y_train, X_val, y_val)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Explainability
    cam = grad_cam(model, X_test[0:1], 'conv5_block3_out')
    plt.imshow(cam)

if __name__ == "__main__":
    main()
