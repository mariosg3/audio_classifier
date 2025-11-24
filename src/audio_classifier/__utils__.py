from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os

def load_data_for_svm(data_dir):

    data = {}

    for split in ["train", "validation", "test"]:
        path = os.path.join(data_dir, f"{split}_data.pt")
        if os.path.exists(path):
            X, y = torch.load(path)
            data[split] = {
                "X": X.numpy(), 
                "y": y.numpy()
            }
            print(f"Loaded {split}: {data[split]['X'].shape}")
        else:
            print(f"Warning: {path} not found.")
    
    return data

def evaluate_svm_model(model, X_test, y_test, class_names=None):

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {acc:.4f}")

    if class_names is None:
        class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'SVM Confusion Matrix (Acc: {acc:.2%})')
    plt.tight_layout()
    plt.savefig('svm_confusion_matrix.png')
    print("Saved confusion matrix to svm_confusion_matrix.png")