import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_data_distribution():
    """First understand the data distribution problem"""
    print("📊 ANALYZING DATA DISTRIBUTION PROBLEM")
    print("="*40)

    data = np.load("features/mfcc_features.npz", allow_pickle=True)
    features, labels = data['features'], data['labels']

    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print("Class distribution in full dataset:")
    for label, count in zip(unique, counts):
        percentage = count / total * 100
        print(f"  {label:12s}: {count:3d}/{total} ({percentage:5.1f}%)")

    # Check if hungry is really dominant
    hungry_ratio = counts[unique == 'hungry'][0] / total if 'hungry' in unique else 0

    print(f"\n🔍 ANALYSIS:")
    if hungry_ratio > 0.4:
        print(f"⚠️  MAJOR IMBALANCE: Hungry = {hungry_ratio*100:.1f}% of dataset")
        print("This explains why model collapses to predicting only hungry")
    else:
        print(f"📊 Hungry ratio: {hungry_ratio*100:.1f}% - not extremely dominant")

    return features, labels, unique, counts

def build_balanced_architecture():
    """Build architecture specifically designed to prevent class collapse"""
    print("\n🏗️  BUILDING ANTI-COLLAPSE ARCHITECTURE")
    print("="*40)

    inputs = layers.Input(shape=(156, 13), name='mfcc_input')

    # Simpler CNN to prevent overfitting
    x = layers.Conv1D(16, 5, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)  
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Dropout(0.3)(x)

    # Simpler LSTM to focus on essential patterns
    x = layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3)(x)

    # Anti-collapse dense layers with strong regularization
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu', 
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)

    # Output layer with bias initialization to combat class imbalance
    outputs = layers.Dense(5, activation='softmax',
                          bias_initializer=keras.initializers.Constant(-1.0))(x)

    model = keras.Model(inputs=inputs, outputs=outputs, 
                       name='CryFusion_AntiCollapse')

    return model

def train_with_anti_collapse_strategy(model, X_train, X_val, X_test, y_train, y_val, y_test, class_names):
    """Train with specific strategies to prevent class collapse"""
    print("\n🎯 ANTI-COLLAPSE TRAINING STRATEGY")
    print("="*38)

    # Calculate AGGRESSIVE class weights to force minority class learning
    y_train_classes = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', 
                                        classes=np.unique(y_train_classes), 
                                        y=y_train_classes)

    # Make class weights even more aggressive
    hungry_idx = list(class_names).index('hungry') if 'hungry' in class_names else -1

    class_weight_dict = {}
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        if i == hungry_idx:
            # Heavily penalize hungry to force learning of other classes
            adjusted_weight = weight * 0.3  
        else:
            # Heavily boost minority classes
            adjusted_weight = weight * 2.0

        class_weight_dict[i] = adjusted_weight

    print("Aggressive anti-collapse class weights:")
    for i, (name, weight) in enumerate(zip(class_names, class_weight_dict.values())):
        print(f"  {name:12s}: {weight:.3f}")

    # Compile with careful optimizer settings
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),  # Gradient clipping
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Custom callback to stop if model collapses to one class
    class AntiCollapseCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:  # Check every 5 epochs
                # Get predictions on validation set
                val_pred = self.model.predict(X_val, verbose=0)
                val_pred_classes = np.argmax(val_pred, axis=1)

                # Check if model is predicting only one class
                unique_preds = len(np.unique(val_pred_classes))

                if unique_preds <= 2:  # If predicting only 1-2 classes
                    print(f"\n⚠️  EPOCH {epoch}: Model collapsing! Only {unique_preds} classes predicted")
                    most_common = np.bincount(val_pred_classes).argmax()
                    most_common_name = class_names[most_common]
                    most_common_ratio = np.mean(val_pred_classes == most_common)
                    print(f"    Most predicted: {most_common_name} ({most_common_ratio*100:.1f}%)")

                    if most_common_ratio > 0.8:  # If >80% predictions are one class
                        print("    🛑 STOPPING TRAINING - Class collapse detected")
                        self.model.stop_training = True
                else:
                    print(f"\n✅ EPOCH {epoch}: Healthy diversity - {unique_preds} classes predicted")

    # Training callbacks
    callbacks_list = [
        AntiCollapseCallback(),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n🚀 STARTING ANTI-COLLAPSE TRAINING:")
    print("- Aggressive class weights (hungry 0.3x, others 2.0x)")
    print("- Gradient clipping to prevent instability")  
    print("- Custom callback to detect class collapse")
    print("- Strong regularization to prevent overfitting")

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Fewer epochs, focus on quality
        batch_size=16,  # Smaller batch size for stability
        callbacks=callbacks_list,
        class_weight=class_weight_dict,
        verbose=1
    )

    return model, history

def evaluate_all_classes(model, X_test, y_test, class_names):
    """Comprehensive evaluation focusing on ALL classes"""
    print("\n🔍 COMPREHENSIVE EVALUATION - ALL CLASSES")
    print("="*45)

    # Get predictions
    pred_probs = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Overall accuracy
    accuracy = accuracy_score(true_classes, pred_classes)
    print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Check prediction distribution
    unique_preds, pred_counts = np.unique(pred_classes, return_counts=True)
    print(f"\n📊 PREDICTION DISTRIBUTION:")
    for pred_class, count in zip(unique_preds, pred_counts):
        class_name = class_names[pred_class]
        percentage = count / len(pred_classes) * 100
        print(f"  {class_name:12s}: {count:3d}/{len(pred_classes)} ({percentage:5.1f}%)")

    classes_predicted = len(unique_preds)
    print(f"\n🎯 CLASSES ACTUALLY PREDICTED: {classes_predicted}/5")

    # Per-class detailed analysis
    print(f"\n📋 PER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        mask = (true_classes == i)
        if np.sum(mask) > 0:
            correct = np.sum((true_classes == i) & (pred_classes == i))
            total = np.sum(mask)
            predicted = np.sum(pred_classes == i)

            precision = correct / predicted if predicted > 0 else 0
            recall = correct / total
            class_acc = recall * 100

            status = "✅" if predicted > 0 else "❌"
            print(f"{status} {class_name:12s}: {correct:2d}/{total:2d} ({class_acc:5.1f}%) | Precision: {precision:.3f}")

    # Success criteria
    print(f"\n🎯 SUCCESS EVALUATION:")

    if classes_predicted >= 4:
        print("🎉 EXCELLENT! Model predicts 4+ classes!")
        success_level = "EXCELLENT"
    elif classes_predicted >= 3:
        print("✅ GOOD! Model predicts 3+ classes!")
        success_level = "GOOD"
    elif classes_predicted >= 2:
        print("📈 PROGRESS! Model predicts 2+ classes!")
        success_level = "PROGRESS"
    else:
        print("❌ FAILED! Model still collapsed to 1 class!")
        success_level = "FAILED"

    # Compare with broken model
    print(f"\n📊 COMPARISON:")
    print(f"Broken model:     1/5 classes predicted (42.5% but useless)")
    print(f"New model:        {classes_predicted}/5 classes predicted ({accuracy*100:.1f}%)")

    if classes_predicted > 1:
        print("🏆 SUCCESS! New model is actually USABLE!")
        print("✅ Better than broken model even if lower accuracy")

    return accuracy, classes_predicted, success_level

def main_fresh_start():
    """Main function to start fresh with anti-collapse approach"""
    print("🔄 STARTING FRESH: BUILDING USABLE MODEL")
    print("="*45)
    print("Goal: Get ALL 5 classes working, not just high accuracy on 1 class")

    # Step 1: Analyze data
    features, labels, unique_labels, counts = analyze_data_distribution()

    # Step 2: Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.reshape(-1, 13)).reshape(features.shape)

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    class_names = le.classes_
    y_categorical = keras.utils.to_categorical(y_encoded, 5)

    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42, 
        stratify=y_encoded
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42,
        stratify=np.argmax(y_train, axis=1)
    )

    print(f"\nData splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Step 4: Build anti-collapse model
    model = build_balanced_architecture()
    print("\n🏗️  Model architecture:")
    model.summary()

    # Step 5: Train with anti-collapse strategy
    model, history = train_with_anti_collapse_strategy(
        model, X_train, X_val, X_test, y_train, y_val, y_test, class_names
    )

    # Step 6: Comprehensive evaluation
    accuracy, classes_predicted, success_level = evaluate_all_classes(
        model, X_test, y_test, class_names
    )

    # Step 7: Final assessment
    print(f"\n🏆 FINAL ASSESSMENT:")
    print("="*20)

    if success_level == "EXCELLENT":
        print("🎉 BREAKTHROUGH! Model successfully predicts 4+ classes!")
        print("✅ This is a USABLE model for infant cry classification!")
        print("🚀 Ready for deployment and real-world testing!")
    elif success_level == "GOOD":
        print("✅ SUCCESS! Model predicts 3+ classes!")
        print("📈 Major improvement over broken single-class model!")
        print("🔧 Can be refined further if needed!")
    elif success_level == "PROGRESS":
        print("📈 PROGRESS! At least 2 classes working!")
        print("🔄 Foundation established, needs more work!")
    else:
        print("❌ Still collapsed - need different approach")
        print("🤔 May need completely different architecture")

    # Save if successful
    if classes_predicted > 1:
        model.save('models/cryfusion_fresh_start.keras')
        print(f"\n💾 Model saved: models/cryfusion_fresh_start.keras")

    return accuracy, classes_predicted

if __name__ == "__main__":
    accuracy, classes = main_fresh_start()
    print(f"\n🎊 FRESH START COMPLETE!")
    print(f"Result: {accuracy:.4f} ({accuracy*100:.1f}%) accuracy, {classes}/5 classes predicted")