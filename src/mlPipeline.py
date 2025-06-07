import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, 
                           f1_score, precision_score, recall_score, 
                           balanced_accuracy_score)
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Δημιουργία φακέλου για αποτελέσματα
import os
os.makedirs('../reports/figures', exist_ok=True)

# --- Συναρτήσεις ---

def load_and_preprocess_data(filepath, target_column, dataset_name=""):
    """
    Φορτώνει και προεπεξεργάζεται τα δεδομένα
    """
    print(f"\n{'='*60}")
    print(f"Φόρτωση {dataset_name} για πρόβλεψη: {target_column}")
    print(f"{'='*60}")
    
    try:
        # Φόρτωση dataset
        df = pd.read_csv(filepath)
        print(f"Διαστάσεις dataset: {df.shape}")
        
        # Έλεγχος αν υπάρχει η στήλη στόχος
        if target_column not in df.columns:
            print(f"Η στήλη '{target_column}' δεν υπάρχει στο dataset.")
            return None, None, None, None, None, True
        
        # Αφαίρεση NaN από τη στήλη στόχο
        initial_rows = len(df)
        df = df.dropna(subset=[target_column])
        if len(df) < initial_rows:
            print(f"Αφαιρέθηκαν {initial_rows - len(df)} γραμμές με NaN στο '{target_column}'")
        
        # Ορισμός features και target
        label_columns = ['Label', 'Traffic Type', 'Traffic Subtype']
        feature_columns = [col for col in df.columns if col not in label_columns]
        
        X = df[feature_columns]
        y_raw = df[target_column]
        
        # Κρατάμε μόνο numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_features].fillna(0)
        
        print(f"Αριθμός features: {X.shape[1]}")
        
        # Έλεγχος για κενό dataset
        if X.empty or len(X) == 0:
            print(f"Κενό dataset μετά την προεπεξεργασία.")
            return None, None, None, None, None, True
        
        # Label encoding
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        # Εμφάνιση πληροφοριών για τις κλάσεις
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"\nΚατανομή κλάσεων:")
        for i, (cls, count) in enumerate(zip(unique_classes, class_counts)):
            original_label = le.inverse_transform([cls])[0]
            print(f"  {original_label}: {count} δείγματα ({count/len(y)*100:.1f}%)")
        
        # Έλεγχος για επαρκή αριθμό κλάσεων
        if len(unique_classes) < 2:
            print(f"Μόνο {len(unique_classes)} κλάση βρέθηκε. Παράλειψη.")
            return None, None, None, None, None, True
        
        # Train-test split με stratification
        test_size = 0.2
        if len(y) < 100:  # Για πολύ μικρά datasets
            test_size = 0.3
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nΔιαχωρισμός δεδομένων:")
        print(f"  Train: {len(X_train)} δείγματα")
        print(f"  Test: {len(X_test)} δείγματα")
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, le, False
        
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση: {str(e)}")
        return None, None, None, None, None, True

def balance_dataset(X_train, y_train, strategy='auto'):
    """
    Εξισορροπεί το dataset με διάφορες στρατηγικές
    """
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    
    # Υπολογισμός imbalance ratio
    imbalance_ratio = max(class_counts) / min(class_counts)
    
    if imbalance_ratio < 2:
        print("  Το dataset είναι ήδη σχετικά ισορροπημένο")
        return X_train, y_train
    
    print(f"  Ανισορροπία {imbalance_ratio:.1f}:1 - Εφαρμογή εξισορρόπησης...")
    
    # Επιλογή στρατηγικής
    min_class_count = min(class_counts)
    
    if min_class_count < 6:  # Πολύ λίγα δείγματα για SMOTE
        print(f"  Χρήση RandomOverSampler (min samples: {min_class_count})")
        sampler = RandomOverSampler(random_state=42)
    else:
        try:
            # Προσπάθεια για SMOTE
            k_neighbors = min(5, min_class_count - 1)
            sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
            print(f"  Χρήση SMOTE με k_neighbors={k_neighbors}")
        except:
            # Fallback σε RandomOverSampler
            print(f"  Fallback σε RandomOverSampler")
            sampler = RandomOverSampler(random_state=42)
    
    # Εφαρμογή sampling
    X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
    
    # Εμφάνιση νέας κατανομής
    new_unique, new_counts = np.unique(y_balanced, return_counts=True)
    print("  Νέα κατανομή:")
    for cls, count in zip(new_unique, new_counts):
        print(f"    Κλάση {cls}: {count} δείγματα")
    
    return X_balanced, y_balanced

def train_svm_model(X_train, X_test, y_train, y_test, label_encoder=None):
    """
    Εκπαιδεύει SVM μοντέλο
    """
    print("\nΕκπαίδευση SVM...")
    start_time = time.time()
    
    # Εξισορρόπηση dataset αν χρειάζεται
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    
    # Επιλογή SVM implementation
    if len(X_train_balanced) > 5000:
        print("  Χρήση LinearSVC για μεγάλο dataset")
        model = LinearSVC(random_state=42, max_iter=2000, dual=False)
    else:
        # Υπολογισμός class weights
        classes = np.unique(y_train_balanced)
        weights = class_weight.compute_class_weight('balanced', 
                                                   classes=classes, 
                                                   y=y_train_balanced)
        class_weights = {i: w for i, w in zip(classes, weights)}
        
        # RBF SVM με probability για μικρά datasets
        model = SVC(kernel='rbf', random_state=42, 
                   class_weight=class_weights, 
                   probability=True)
    
    # Εκπαίδευση
    model.fit(X_train_balanced, y_train_balanced)
    
    # Πρόβλεψη
    y_pred = model.predict(X_test)
    
    # Χρόνος εκπαίδευσης
    training_time = time.time() - start_time
    print(f"  ✓ Ολοκληρώθηκε σε {training_time:.2f} δευτερόλεπτα")
    
    return model, y_pred, training_time

def train_mlp_model(X_train, X_test, y_train, y_test, label_encoder=None):
    """
    Εκπαιδεύει Neural Network μοντέλο
    """
    print("\nΕκπαίδευση Neural Network...")
    start_time = time.time()
    
    # Εξισορρόπηση dataset
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)
    
    # Αρχιτεκτονική NN ανάλογα με το μέγεθος του dataset
    n_samples = len(X_train_balanced)
    n_features = X_train_balanced.shape[1]
    
    if n_samples < 1000:
        hidden_layers = (50, 25)
        max_iter = 1000
    elif n_samples < 10000:
        hidden_layers = (100, 50)
        max_iter = 500
    else:
        hidden_layers = (200, 100, 50)
        max_iter = 300
    
    print(f"  Αρχιτεκτονική: {n_features} → {' → '.join(map(str, hidden_layers))} → output")
    
    # Δημιουργία και εκπαίδευση μοντέλου
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    
    # Εκπαίδευση
    model.fit(X_train_balanced, y_train_balanced)
    
    # Πρόβλεψη
    y_pred = model.predict(X_test)
    
    # Χρόνος εκπαίδευσης
    training_time = time.time() - start_time
    print(f"  Ολοκληρώθηκε σε {training_time:.2f} δευτερόλεπτα")
    print(f"  Iterations: {model.n_iter_}")
    
    return model, y_pred, training_time

def evaluate_model(y_test, y_pred, label_encoder=None, model_name=""):
    """
    Αξιολογεί το μοντέλο και επιστρέφει μετρικές
    """
    print(f"\nΑξιολόγηση {model_name}:")
    
    # Βασικές μετρικές
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Υπολογισμός F1 score
    unique_classes = np.unique(y_test)
    if len(unique_classes) == 2:
        # Binary classification
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=1)
        precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
        recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
    else:
        # Multiclass
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
    
    # Εμφάνιση αποτελεσμάτων
    print(f"  • Accuracy: {accuracy:.4f}")
    print(f"  • Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  • F1-Score: {f1:.4f}")
    print(f"  • Precision: {precision:.4f}")
    print(f"  • Recall: {recall:.4f}")
    
    # Classification report
    if label_encoder is not None:
        print("\nΛεπτομερής αναφορά ανά κλάση:")
        target_names = [str(label) for label in label_encoder.classes_]
        print(classification_report(y_test, y_pred, 
                                  target_names=target_names,
                                  digits=3))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, labels, title, save_path=None):
    """
    Δημιουργεί heatmap για confusion matrix
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_results_visualization(results_df):
    """
    Δημιουργεί συγκριτικά γραφήματα για τα αποτελέσματα
    """
    # FIGURE 1: Μετρικές Απόδοσης
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle('Μετρικές Απόδοσης Μοντέλων', fontsize=16)
    
    # 1. Accuracy comparison by Dataset
    ax = axes1[0]
    pivot_acc = results_df.pivot_table(index='Dataset', columns='Model', 
                                      values='Accuracy', aggfunc='mean')
    pivot_acc.plot(kind='bar', ax=ax)
    ax.set_title('Accuracy ανά Dataset', fontsize=14)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    ax.legend(title='Model')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 2. F1-Score comparison by Dataset
    ax = axes1[1]
    pivot_f1 = results_df.pivot_table(index='Dataset', columns='Model', 
                                     values='F1-Score', aggfunc='mean')
    pivot_f1.plot(kind='bar', ax=ax)
    ax.set_title('F1-Score ανά Dataset', fontsize=14)
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0, 1.1)
    ax.legend(title='Model')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Balanced Accuracy comparison
    ax = axes1[2]
    pivot_bal = results_df.pivot_table(index='Dataset', columns='Model', 
                                      values='Balanced Accuracy', aggfunc='mean')
    pivot_bal.plot(kind='bar', ax=ax)
    ax.set_title('Balanced Accuracy ανά Dataset', fontsize=14)
    ax.set_ylabel('Balanced Accuracy')
    ax.set_ylim(0, 1.1)
    ax.legend(title='Model')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('../reports/figures/model_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # FIGURE 2: Συγκριτική Ανάλυση
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Συγκριτική Ανάλυση Μοντέλων', fontsize=16)
    
    # 1. Performance by Target
    ax = axes2[0]
    for target in results_df['Target'].unique():
        target_data = results_df[results_df['Target'] == target]
        avg_f1 = target_data.groupby('Model')['F1-Score'].mean()
        x_pos = np.arange(len(avg_f1))
        ax.bar(x_pos + (0.35 if target == 'Label' else -0.35)/2, 
               avg_f1.values, 0.35, label=target)
    ax.set_xticks(np.arange(len(avg_f1)))
    ax.set_xticklabels(avg_f1.index)
    ax.set_title('Μέση F1-Score ανά Target', fontsize=14)
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Training Time comparison
    ax = axes2[1]
    pivot_time = results_df.pivot_table(index='Dataset', columns='Model', 
                                       values='Training Time', aggfunc='mean')
    pivot_time.plot(kind='bar', ax=ax, logy=True)
    ax.set_title('Χρόνος Εκπαίδευσης (log scale)', fontsize=14)
    ax.set_ylabel('Time (seconds)')
    ax.legend(title='Model')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Best Models Summary
    ax = axes2[2]
    best_by_target = results_df.loc[results_df.groupby('Target')['F1-Score'].idxmax()]
    
    y_pos = np.arange(len(best_by_target))
    bars = ax.barh(y_pos, best_by_target['F1-Score'])
    
    # Χρωματισμός ανάλογα με το μοντέλο
    colors = ['#1f77b4' if 'SVM' in model else '#ff7f0e' for model in best_by_target['Model']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['Target']}\n{row['Dataset']}\n{row['Model']}" 
                        for _, row in best_by_target.iterrows()], fontsize=10)
    ax.set_xlabel('F1-Score')
    ax.set_xlim(0, 1.1)
    ax.set_title('Καλύτερα Μοντέλα ανά Target', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Προσθήκη τιμών στις μπάρες
    for i, (_, row) in enumerate(best_by_target.iterrows()):
        ax.text(row['F1-Score'] + 0.01, i, f"{row['F1-Score']:.3f}", 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../reports/figures/model_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- ΚΥΡΙΑ ΕΚΤΕΛΕΣΗ ---

def main():
    """
    Κύρια συνάρτηση εκτέλεσης
    """
    print("="*80)
    print("ΕΡΩΤΗΜΑ 3: ΕΚΠΑΙΔΕΥΣΗ ΚΑΙ ΑΞΙΟΛΟΓΗΣΗ ΜΟΝΤΕΛΩΝ ΤΑΞΙΝΟΜΗΣΗΣ")
    print("="*80)
    
    # Datasets που δημιουργήθηκαν στο Ερώτημα 2
    datasets_info = {
        "Balanced_Sampled": "../data/processed/dataset_1_balanced_sampled.csv",
        "KMeans": "../data/processed/dataset_2_kmeans.csv",
        "DBSCAN": "../data/processed/dataset_3_dbscan.csv"
    }
    
    # Λίστα για αποθήκευση αποτελεσμάτων
    all_results = []
    
    # Επεξεργασία κάθε dataset
    for dataset_name, dataset_path in datasets_info.items():
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*80}")
        
        # --- TASK 1: Πρόβλεψη Label (Benign/Malicious) ---
        print(f"\nTASK 1: Πρόβλεψη Label (Benign/Malicious)")
        print("-"*60)
        
        X_train, X_test, y_train, y_test, le_label, skip = load_and_preprocess_data(
            dataset_path, 'Label', dataset_name
        )
        
        if not skip and X_train is not None:
            # SVM
            svm_model, y_pred_svm, svm_time = train_svm_model(
                X_train, X_test, y_train, y_test, le_label
            )
            svm_metrics = evaluate_model(y_test, y_pred_svm, le_label, "SVM")
            
            # Αποθήκευση αποτελεσμάτων
            all_results.append({
                'Dataset': dataset_name,
                'Target': 'Label',
                'Model': 'SVM',
                'Accuracy': svm_metrics['accuracy'],
                'Balanced Accuracy': svm_metrics['balanced_accuracy'],
                'F1-Score': svm_metrics['f1_score'],
                'Precision': svm_metrics['precision'],
                'Recall': svm_metrics['recall'],
                'Training Time': svm_time
            })
            
            # Neural Network
            mlp_model, y_pred_mlp, mlp_time = train_mlp_model(
                X_train, X_test, y_train, y_test, le_label
            )
            mlp_metrics = evaluate_model(y_test, y_pred_mlp, le_label, "Neural Network")
            
            # Αποθήκευση αποτελεσμάτων
            all_results.append({
                'Dataset': dataset_name,
                'Target': 'Label',
                'Model': 'Neural Network',
                'Accuracy': mlp_metrics['accuracy'],
                'Balanced Accuracy': mlp_metrics['balanced_accuracy'],
                'F1-Score': mlp_metrics['f1_score'],
                'Precision': mlp_metrics['precision'],
                'Recall': mlp_metrics['recall'],
                'Training Time': mlp_time
            })
            
            # Confusion Matrices
            if le_label is not None:
                labels = le_label.classes_
                plot_confusion_matrix(
                    svm_metrics['confusion_matrix'], 
                    labels,
                    f'Confusion Matrix - {dataset_name} - Label - SVM',
                    f'../reports/figures/cm_{dataset_name}_label_svm.png'
                )
                plot_confusion_matrix(
                    mlp_metrics['confusion_matrix'], 
                    labels,
                    f'Confusion Matrix - {dataset_name} - Label - MLP',
                    f'../reports/figures/cm_{dataset_name}_label_mlp.png'
                )
        
        # --- TASK 2: Πρόβλεψη Traffic Type ---
        print(f"\nTASK 2: Πρόβλεψη Traffic Type")
        print("-"*60)
        
        X_train, X_test, y_train, y_test, le_traffic, skip = load_and_preprocess_data(
            dataset_path, 'Traffic Type', dataset_name
        )
        
        if not skip and X_train is not None:
            # SVM
            svm_model, y_pred_svm, svm_time = train_svm_model(
                X_train, X_test, y_train, y_test, le_traffic
            )
            svm_metrics = evaluate_model(y_test, y_pred_svm, le_traffic, "SVM")
            
            # Αποθήκευση αποτελεσμάτων
            all_results.append({
                'Dataset': dataset_name,
                'Target': 'Traffic Type',
                'Model': 'SVM',
                'Accuracy': svm_metrics['accuracy'],
                'Balanced Accuracy': svm_metrics['balanced_accuracy'],
                'F1-Score': svm_metrics['f1_score'],
                'Precision': svm_metrics['precision'],
                'Recall': svm_metrics['recall'],
                'Training Time': svm_time
            })
            
            # Neural Network
            mlp_model, y_pred_mlp, mlp_time = train_mlp_model(
                X_train, X_test, y_train, y_test, le_traffic
            )
            mlp_metrics = evaluate_model(y_test, y_pred_mlp, le_traffic, "Neural Network")
            
            # Αποθήκευση αποτελεσμάτων
            all_results.append({
                'Dataset': dataset_name,
                'Target': 'Traffic Type',
                'Model': 'Neural Network',
                'Accuracy': mlp_metrics['accuracy'],
                'Balanced Accuracy': mlp_metrics['balanced_accuracy'],
                'F1-Score': mlp_metrics['f1_score'],
                'Precision': mlp_metrics['precision'],
                'Recall': mlp_metrics['recall'],
                'Training Time': mlp_time
            })
    
    # --- ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ---
    print(f"\n{'='*80}")
    print("ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
    print(f"{'='*80}")
    
    # Δημιουργία DataFrame με αποτελέσματα
    results_df = pd.DataFrame(all_results)
    
    # Εμφάνιση πίνακα αποτελεσμάτων
    print("\nΠίνακας Αποτελεσμάτων:")
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Αποθήκευση αποτελεσμάτων σε CSV
    results_df.to_csv('../reports/model_results.csv', index=False)
    print("\nΑποτελέσματα αποθηκεύτηκαν στο: ../reports/model_results.csv")
    
    # --- ΑΝΑΛΥΣΗ ΚΑΛΥΤΕΡΩΝ ΜΟΝΤΕΛΩΝ ---
    print(f"\n{'='*80}")
    print("ΑΝΑΛΥΣΗ ΚΑΛΥΤΕΡΩΝ ΜΟΝΤΕΛΩΝ")
    print(f"{'='*80}")
    
    # Καλύτερο μοντέλο για Label prediction
    label_results = results_df[results_df['Target'] == 'Label']
    if not label_results.empty:
        best_label_idx = label_results['F1-Score'].idxmax()
        best_label = label_results.loc[best_label_idx]
        
        print(f"\nΚαλύτερο μοντέλο για πρόβλεψη Label (Benign/Malicious):")
        print(f" • Dataset: {best_label['Dataset']}")
        print(f" • Model: {best_label['Model']}")
        print(f" • F1-Score: {best_label['F1-Score']:.4f}")
        print(f" • Balanced Accuracy: {best_label['Balanced Accuracy']:.4f}")
        print(f" • Training Time: {best_label['Training Time']:.2f}s")
    
    # Καλύτερο μοντέλο για Traffic Type prediction
    traffic_results = results_df[results_df['Target'] == 'Traffic Type']
    if not traffic_results.empty:
        best_traffic_idx = traffic_results['F1-Score'].idxmax()
        best_traffic = traffic_results.loc[best_traffic_idx]
        
        print(f"\nΚαλύτερο μοντέλο για πρόβλεψη Traffic Type:")
        print(f" • Dataset: {best_traffic['Dataset']}")
        print(f" • Model: {best_traffic['Model']}")
        print(f" • F1-Score: {best_traffic['F1-Score']:.4f}")
        print(f" • Balanced Accuracy: {best_traffic['Balanced Accuracy']:.4f}")
        print(f" • Training Time: {best_traffic['Training Time']:.2f}s")
    
    # --- ΣΥΓΚΡΙΤΙΚΗ ΑΝΑΛΥΣΗ ---
    print(f"\n{'='*80}")
    print("ΣΥΓΚΡΙΤΙΚΗ ΑΝΑΛΥΣΗ")
    print(f"{'='*80}")
    
    # Μέσοι όροι ανά μοντέλο
    print("\nΜέση απόδοση ανά μοντέλο:")
    model_avg = results_df.groupby('Model')[['F1-Score', 'Balanced Accuracy']].mean()
    print(model_avg.round(4))
    
    # Μέσοι όροι ανά dataset
    print("\nΜέση απόδοση ανά dataset:")
    dataset_avg = results_df.groupby('Dataset')[['F1-Score', 'Balanced Accuracy']].mean()
    print(dataset_avg.round(4))
    
    # Μέσοι όροι ανά target
    print("\nΜέση απόδοση ανά target:")
    target_avg = results_df.groupby('Target')[['F1-Score', 'Balanced Accuracy']].mean()
    print(target_avg.round(4))
    
    # Δημιουργία γραφημάτων
    create_results_visualization(results_df)
    
    print("\nΗ ανάλυση ολοκληρώθηκε επιτυχώς!")

if __name__ == "__main__":
    main()