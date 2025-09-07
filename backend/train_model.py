import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

def build_optimized_baseline_model():
    """
    Your proven architecture with optimizations for original dataset
    """
    inputs = layers.Input(shape=(156, 13), name='mfcc_input')

    # CNN layers - same as working model but slightly optimized
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)  # Slightly less dropout

    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # LSTM layers - same as working
    x = layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)  # Less dropout
    x = layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3)(x)

    # Dense layers with optimization
    x = layers.Dropout(0.4)(x)  # Less aggressive dropout
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(5, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='CryFusion_Optimized_Baseline')

    # Optimized learning rate
    optimizer = optimizers.Adam(learning_rate=0.0005)  # Slightly higher than before
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return model

def train_optimized_baseline():
    """
    Train optimized model on ORIGINAL dataset to recover and exceed baseline
    """
    print("🔄 BACK TO BASELINE + OPTIMIZATION STRATEGY")
    print("="*50)
    print("Goal 1: Recover 43.6% baseline performance")
    print("Goal 2: Push to 50-60% through training optimization")

    # Load ORIGINAL dataset (not augmented)
    orig_path = Path("features/mfcc_features.npz")

    if not orig_path.exists():
        print("❌ Original MFCC features not found!")
        return

    data = np.load(orig_path, allow_pickle=True)
    features, labels = data['features'], data['labels']

    print(f"✅ Using ORIGINAL dataset: {features.shape}")
    print("This is the dataset that achieved 43.6% - we know it works!")

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nOriginal class distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label:12s}: {count:3d} samples")

    # Data preparation with normalization
    print("\n🔧 Optimizing data preparation...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(-1, 13)).reshape(features.shape)

    # Label encoding
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_
    y = keras.utils.to_categorical(labels_encoded, 5)

    # Optimized class weights (learned from previous experiments)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_encoded), y=labels_encoded)

    # Fine-tuned class weights based on what we learned
    class_weight_dict = {}
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        if name == 'hungry':
            adjusted_weight = weight * 0.7  # Reduce hungry dominance
        elif name == 'discomfort':
            adjusted_weight = weight * 0.9  # Slight reduction (was over-predicted)
        else:
            adjusted_weight = weight * 1.1  # Slight boost for others

        class_weight_dict[i] = adjusted_weight

    print(f"\nOptimized class weights:")
    for i, (name, weight) in enumerate(zip(class_names, class_weight_dict.values())):
        print(f"  {name:12s}: {weight:.2f}")

    # Data splits
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, y, test_size=0.2, random_state=42, stratify=labels_encoded
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, 
        stratify=np.argmax(y_train, axis=1)
    )

    print(f"\nData splits:")
    print(f"Training: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    # Build optimized model
    model = build_optimized_baseline_model()
    print("\n🏗️  OPTIMIZED MODEL ARCHITECTURE:")
    model.summary()

    # Optimized training callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,  # Very patient
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # Only stop if really not improving
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # Less aggressive LR reduction
            patience=15,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath='models/cryfusion_optimized_baseline.keras',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n🚀 OPTIMIZED TRAINING STRATEGY:")
    print("- Higher learning rate for better convergence")
    print("- Less aggressive dropout to retain information")
    print("- Very patient early stopping")
    print("- Balanced but not extreme class weights")
    print("- Longer training time allowed")

    # Train with optimization
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,  # More epochs
        batch_size=32,
        callbacks=callbacks_list,
        class_weight=class_weight_dict,
        verbose=1
    )

    print("\n✅ OPTIMIZED TRAINING COMPLETED!")

    # Comprehensive evaluation
    print("\n📊 EVALUATING OPTIMIZED BASELINE...")

    test_metrics = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_acc, test_prec, test_rec = test_metrics

    print(f"\n🎯 OPTIMIZED BASELINE RESULTS:")
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")

    # Performance analysis
    previous_best = 0.436
    improvement = test_acc - previous_best

    print(f"\n📈 PERFORMANCE ANALYSIS:")
    print(f"Previous best:  {previous_best:.4f} ({previous_best*100:.1f}%)")
    print(f"Current result: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"Change:         {improvement:.4f} ({improvement*100:.1f}%)")

    # Success evaluation
    if test_acc >= 0.50:
        print("\n🎉 OUTSTANDING! 50%+ accuracy achieved!")
        print("✅ Target reached through optimization alone!")
        status = "TARGET_ACHIEVED"
    elif test_acc >= 0.47:
        print("\n🏆 EXCELLENT! Close to 50% target!")
        print("📈 Great progress - ensemble might push over 50%")
        status = "CLOSE_TO_TARGET"
    elif test_acc >= 0.44:
        print("\n✅ GOOD! Baseline recovered and improved!")
        print("📈 Ready for next phase of optimization")
        status = "BASELINE_EXCEEDED"
    elif test_acc >= 0.40:
        print("\n📈 PROGRESS! Close to baseline recovery")
        print("🔧 Need slight tuning to reach previous best")
        status = "NEAR_BASELINE"
    else:
        print("\n⚠️  Below baseline - need investigation")
        status = "BELOW_BASELINE"

    # Detailed per-class analysis
    print(f"\n📋 DETAILED PER-CLASS ANALYSIS:")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    for i, class_name in enumerate(class_names):
        class_mask = (y_true_classes == i)
        pred_mask = (y_pred_classes == i)

        if np.sum(class_mask) > 0:
            correct = np.sum((y_true_classes == i) & (y_pred_classes == i))
            total = np.sum(class_mask)
            predicted = np.sum(pred_mask)
            accuracy = correct / total * 100

            print(f"{class_name:12s}: {correct:2d}/{total:2d} correct ({accuracy:5.1f}%) | Predicted {predicted:2d} times")

    # Save model
    model.save('models/cryfusion_final_optimized.keras')
    print(f"\n💾 Optimized model saved!")

    # Next steps recommendation
    print(f"\n🎯 NEXT STEPS FOR {status}:")
    if status == "TARGET_ACHIEVED":
        print("🎊 SUCCESS! You've reached 50%+ accuracy!")
        print("Optional: Try ensemble for even higher performance")
    elif status == "CLOSE_TO_TARGET":
        print("🚀 Try small ensemble (2-3 models) to push over 50%")
        print("Or try gentle data augmentation (very conservative)")
    elif status == "BASELINE_EXCEEDED":
        print("📈 Good foundation! Options:")
        print("1. Try longer training (200+ epochs)")
        print("2. Small ensemble (2 models)")
        print("3. Conservative hyperparameter tuning")
    else:
        print("🔧 Focus on hyperparameter optimization")
        print("1. Learning rate adjustment")
        print("2. Architecture modifications")
        print("3. Different regularization")

    return test_acc, status

if __name__ == "__main__":
    accuracy, status = train_optimized_baseline()
    print(f"\n🎊 BASELINE OPTIMIZATION COMPLETE!")
    print(f"Result: {accuracy:.4f} ({accuracy*100:.1f}% accuracy)")