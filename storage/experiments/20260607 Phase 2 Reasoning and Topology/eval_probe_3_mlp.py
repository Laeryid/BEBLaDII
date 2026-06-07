import os
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

def main():
    features_path = r"C:\Experiments\BEBLaDII\storage\experiments\eval\snli_features.pt"
    
    if not os.path.exists(features_path):
        print(f"Error: {features_path} not found. Run eval_probe_1_extract.py first!")
        return

    print(f"Loading features from {features_path}...")
    data = torch.load(features_path, map_location="cpu")
    
    X_train = data["X_train"].numpy()
    y_train = data["y_train"].numpy()
    X_test = data["X_test"].numpy()
    y_test = data["y_test"].numpy()
    
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    print("\nTraining Non-Linear MLP Probe...")
    start_time = time.time()
    
    # Крошечный MLP: 1 скрытый слой на 512 нейронов
    # early_stopping=True позволяет избежать переобучения на обучающей выборке
    clf = MLPClassifier(
        hidden_layer_sizes=(512,), 
        max_iter=500, 
        early_stopping=True, 
        verbose=True,
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds.")
    
    print("\nEvaluating on test set...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print("=" * 50)
    print(f"Non-Linear MLP Probing Accuracy: {acc * 100:.2f}%")
    print("=" * 50)
    
    print("\nDetailed Report:")
    target_names = ["Entailment (0)", "Neutral (1)", "Contradiction (2)"]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
if __name__ == "__main__":
    main()
