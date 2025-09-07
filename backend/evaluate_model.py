import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

def load_ensemble_models(path_prefix='models/cryfusion_ensemble_target'):
    """Load trained ensemble models"""
    models = []
    scalers = []

    # Load metadata
    with open(f'{path_prefix}_metadata.json', 'r') as f:
        metadata = json.load(f)

    n_models = metadata['n_models']

    for i in range(n_models):
        model_path = f'{path_prefix}_model_{i}.keras'
        if Path(model_path).exists():
            model = tf.keras.models.load_model(model_path)
            models.append(model)

            # Create dummy scaler (will be replaced with actual scalers if saved)
            scaler = StandardScaler()
            scalers.append(scaler)

    return models, scalers, metadata

def predict_ensemble(models, scalers, X):
    """Make ensemble predictions"""
    predictions = []

    for i, (model, scaler) in enumerate(zip(models, scalers)):
        # Fit scaler on current data (since we didn't save scalers)
        X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        pred = model.predict(X_scaled, verbose=0)
        predictions.append(pred)

    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

def evaluate_final_performance():
    """Comprehensive evaluation of final model performance"""
    print("🎯 FINAL CRYFUSION PERFORMANCE EVALUATION")
    print("="*60)
    print("Checking if 50-60% accuracy target was achieved...")

    # Try to load ensemble models first
    ensemble_path = "models/cryfusion_ensemble_target"
    if Path(f"{ensemble_path}_metadata.json").exists():
        print("\n✅ Found ensemble models - evaluating ensemble performance")

        # Load ensemble
        try:
            models, scalers, metadata = load_ensemble_models(ensemble_path)
            print(f"Loaded {len(models)} ensemble models")

            # Load test data (use augmented if available, otherwise original)
            if Path("features/mfcc_features_augmented.npz").exists():
                print("Using augmented dataset for evaluation...")
                data_path = "features/mfcc_features_augmented.npz"
            else:
                print("Using original dataset for evaluation...")
                data_path = "features/mfcc_features.npz"

            data = np.load(data_path, allow_pickle=True)
            features, labels = data['features'], data['labels']

            # Prepare data
            label_encoder = LabelEncoder()
            labels_encoded = label_encoder.fit_transform(labels)
            class_names = label_encoder.classes_

            # Same test split as training
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels_encoded, test_size=0.15, random_state=42, stratify=labels_encoded
            )

            print(f"Test set: {len(X_test)} samples")
            print(f"Classes: {list(class_names)}")

            # Make ensemble predictions
            ensemble_pred = predict_ensemble(models, scalers, X_test)
            ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)

            # Calculate accuracy
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred_classes)

            print(f"\n🏆 ENSEMBLE RESULTS:")
            print(f"Final Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.1f}%)")

            # Target achievement check
            if ensemble_accuracy >= 0.60:
                print("\n🎉 OUTSTANDING! 60%+ accuracy achieved!")
                print("You've exceeded the target - this is research-level performance!")
                status = "OUTSTANDING"
            elif ensemble_accuracy >= 0.55:
                print("\n🏆 EXCELLENT! 55%+ accuracy achieved!")
                print("Target significantly exceeded - fantastic work!")
                status = "EXCELLENT"
            elif ensemble_accuracy >= 0.50:
                print("\n✅ SUCCESS! 50%+ accuracy target achieved!")
                print("Your CryFusion system now meets the performance requirements!")
                status = "SUCCESS"
            elif ensemble_accuracy >= 0.45:
                print("\n📈 VERY CLOSE! 45%+ accuracy - almost there!")
                print("Consider running ensemble training longer or with more augmentation.")
                status = "CLOSE"
            else:
                print("\n📊 GOOD PROGRESS! Continue with optimization strategies.")
                status = "PROGRESS"

            # Improvement analysis
            original_acc = 0.276  # From earlier results
            improvement = (ensemble_accuracy - original_acc) / original_acc * 100
            print(f"\n📊 IMPROVEMENT ANALYSIS:")
            print(f"Original accuracy: {original_acc:.4f} ({original_acc*100:.1f}%)")
            print(f"Final accuracy:    {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.1f}%)")
            print(f"Total improvement: {improvement:.1f}%")

            # Per-class performance
            print(f"\n📋 DETAILED PER-CLASS PERFORMANCE:")
            class_report = classification_report(
                y_test, ensemble_pred_classes,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )

            for i, class_name in enumerate(class_names):
                if str(i) in class_report:
                    metrics = class_report[str(i)]
                    support = int(metrics['support'])
                    print(f"{class_name:12s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f} (n={support})")

            # Confusion matrix
            cm = confusion_matrix(y_test, ensemble_pred_classes)
            print(f"\n📊 PREDICTION DISTRIBUTION:")
            print(f"True\\Pred   ", end="")
            for name in class_names:
                print(f"{name[:4]:>5s}", end="")
            print()

            for i, true_name in enumerate(class_names):
                print(f"{true_name[:10]:10s}", end="")
                for j in range(len(class_names)):
                    print(f"{cm[i,j]:5d}", end="")
                print()

            return ensemble_accuracy, status

        except Exception as e:
            print(f"❌ Error evaluating ensemble: {e}")
            print("Falling back to single model evaluation...")

    # Fallback: Check single best model
    print("\n🔍 Checking single model performance...")

    best_models = [
        "models/cryfusion_improved.keras",
        "models/cryfusion_final.keras", 
        "models/cryfusion_mfcc_cnn_lstm.keras"
    ]

    best_accuracy = 0
    best_model_name = None

    for model_path in best_models:
        if Path(model_path).exists():
            try:
                print(f"Evaluating {model_path}...")

                # Load model and data  
                model = tf.keras.models.load_model(model_path)

                if Path("features/mfcc_features_augmented.npz").exists():
                    data_path = "features/mfcc_features_augmented.npz"
                else:
                    data_path = "features/mfcc_features.npz"

                data = np.load(data_path, allow_pickle=True)
                features, labels = data['features'], data['labels']

                # Prepare data
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)

                label_encoder = LabelEncoder()
                labels_encoded = label_encoder.fit_transform(labels)

                X_train, X_test, y_train, y_test = train_test_split(
                    features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
                )

                # Predict
                y_pred = model.predict(X_test, verbose=0)
                y_pred_classes = np.argmax(y_pred, axis=1)

                accuracy = accuracy_score(y_test, y_pred_classes)
                print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model_path

            except Exception as e:
                print(f"  Error: {e}")

    if best_accuracy > 0:
        print(f"\n🏆 BEST SINGLE MODEL PERFORMANCE:")
        print(f"Model: {best_model_name}")
        print(f"Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")

        if best_accuracy >= 0.50:
            print("✅ TARGET ACHIEVED with single model!")
            return best_accuracy, "SUCCESS"
        else:
            print("📈 Good progress - ensemble training recommended for target achievement")
            return best_accuracy, "PROGRESS"

    print("\n❌ No models found for evaluation")
    return 0, "NO_MODELS"

def provide_recommendations(accuracy, status):
    """Provide specific recommendations based on results"""
    print(f"\n🎯 RECOMMENDATIONS BASED ON {status} STATUS:")
    print("="*50)

    if status == "OUTSTANDING" or status == "EXCELLENT" or status == "SUCCESS":
        print("🎊 CONGRATULATIONS! Your CryFusion system is ready for deployment!")
        print("\nWhat you can do now:")
        print("✅ Deploy your model for real-world use")
        print("✅ Create a user interface for parents/caregivers")
        print("✅ Test with new audio samples")
        print("✅ Consider publishing your results")

    elif status == "CLOSE":
        print("🚀 You're very close! Try these final optimizations:")
        print("1. Run ensemble training with more epochs (100-120)")
        print("2. Create more augmented data (target 400 samples per class)")
        print("3. Use advanced features (30 features instead of 13)")
        print("4. Fine-tune class weights further")

    elif status == "PROGRESS":
        print("📈 Continue with the optimization plan:")
        print("1. MUST DO: python create_augmented_dataset.py")
        print("2. MUST DO: python train_ensemble_target.py")
        print("3. OPTIONAL: python create_advanced_features.py")
        print("4. Expected result: 50-60% accuracy achievement")

    else:
        print("🔧 Start with the optimization pipeline:")
        print("1. Run data augmentation first")
        print("2. Then train ensemble models")
        print("3. This should get you to 50-60% target")

if __name__ == "__main__":
    accuracy, status = evaluate_final_performance()
    provide_recommendations(accuracy, status)