import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Path του αρχείου
file_path = '../data/raw/data.csv'

print("="*60)
print("ΕΡΩΤΗΜΑ 2: ΜΕΙΩΣΗ ΔΙΑΣΤΑΤΙΚΟΤΗΤΑΣ ΚΑΙ ΔΕΙΓΜΑΤΟΛΗΨΙΑ")
print("="*60)

# ΜΕΡΟΣ 1: ΜΕΙΩΣΗ ΔΙΑΣΤΑΤΙΚΟΤΗΤΑΣ (ΣΤΗΛΩΝ)
print("\n=== ΜΕΡΟΣ 1: ΜΕΙΩΣΗ ΔΙΑΣΤΑΤΙΚΟΤΗΤΑΣ ===\n")

# Διαβάζουμε ένα δείγμα για να αναλύσουμε τις στήλες
sample_df = pd.read_csv(file_path, nrows=10000)
print(f"Αρχικός αριθμός στηλών: {len(sample_df.columns)}")

# 1. Αφαίρεση στηλών με χαμηλή πληροφορία
print("\n1. Αφαίρεση στηλών με χαμηλή διακύμανση και μοναδικές τιμές...")

# Λίστα στηλών που θα αφαιρέσουμε
columns_to_remove = []

# Αφαιρούμε το Flow ID (μοναδικός identifier χωρίς predictive value)
columns_to_remove.append('Flow ID')
columns_to_remove.extend(['Src IP', 'Dst IP'])

# Αφαιρούμε το Timestamp για γενίκευση
columns_to_remove.append('Timestamp')

# Βρίσκουμε στήλες με μηδενική ή πολύ χαμηλή διακύμανση
numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if sample_df[col].std() < 0.01:
        columns_to_remove.append(col)

# Αφαιρούμε URG flags που είναι σχεδόν πάντα 0
urg_cols = [col for col in sample_df.columns if 'URG' in col]
columns_to_remove.extend(urg_cols)

# Αφαιρούμε ECE και CWR flags που είναι σπάνια
rare_flags = ['ECE Flag Count', 'CWR Flag Count']
columns_to_remove.extend([col for col in rare_flags if col in sample_df.columns])

# Καθαρίζουμε τα duplicates
columns_to_remove = list(set(columns_to_remove))
print(f"Στήλες που θα αφαιρεθούν: {len(columns_to_remove)}")

# 2. Δημιουργία νέων συνδυαστικών features
print("\n2. Δημιουργία συνδυαστικών χαρακτηριστικών...")

def create_combined_features(df):
    # Λόγος Forward/Backward πακέτων
    df['Fwd_Bwd_Packet_Ratio'] = (df['Total Fwd Packet'] + 1) / (df['Total Bwd packets'] + 1)
    
    # Συνολικά πακέτα
    df['Total_Packets'] = df['Total Fwd Packet'] + df['Total Bwd packets']
    
    # Συνολικό μέγεθος
    df['Total_Bytes'] = df['Total Length of Fwd Packet'] + df['Total Length of Bwd Packet']
    
    # Μέσο μέγεθος πακέτου (συνολικά)
    df['Avg_Packet_Size_Total'] = df['Total_Bytes'] / (df['Total_Packets'] + 1)
    
    # Συμμετρία ροής
    df['Flow_Symmetry'] = 1 - abs(df['Total Fwd Packet'] - df['Total Bwd packets']) / (df['Total_Packets'] + 1)
    
    # Ενεργότητα ροής (πακέτα ανά δευτερόλεπτο)
    df['Flow_Activity'] = df['Total_Packets'] / (df['Flow Duration'] / 1e6 + 1)  # Duration σε seconds
    
    return df

# 3. Επιλογή τελικών στηλών
print("\n3. Επιλογή τελικών στηλών για το μειωμένο dataset...")

# Στήλες που θα κρατήσουμε (βασισμένες στην ανάλυση του Ερωτήματος 1)
important_features = [
    # Βασικά identifiers
    'Src Port', 'Dst Port', 'Protocol',
    
    # Χρονικά χαρακτηριστικά
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    
    # Μεγέθη πακέτων
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Packet Length Mean', 'Packet Length Std',
    
    # Ρυθμοί
    'Flow Bytes/s', 'Flow Packets/s', 
    'Fwd Packets/s', 'Bwd Packets/s',
    
    # Flags
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 
    'PSH Flag Count', 'ACK Flag Count',
    
    # Window sizes
    'FWD Init Win Bytes', 'Bwd Init Win Bytes',
    
    # Active/Idle times
    'Active Mean', 'Active Std', 'Idle Mean', 'Idle Std',
    
    # Labels
    'Label', 'Traffic Type', 'Traffic Subtype'
]

# Προσθέτουμε και τα νέα combined features
new_features = [
    'Fwd_Bwd_Packet_Ratio', 'Total_Packets', 'Total_Bytes',
    'Avg_Packet_Size_Total', 'Flow_Symmetry', 'Flow_Activity'
]

all_features = important_features + new_features

print(f"Τελικός αριθμός στηλών: {len(all_features)}")
print(f"Μείωση: {len(sample_df.columns)} -> {len(all_features)} στήλες")

# ΜΕΡΟΣ 2: ΜΕΙΩΣΗ ΓΡΑΜΜΩΝ
print("\n=== ΜΕΡΟΣ 2: ΜΕΙΩΣΗ ΓΡΑΜΜΩΝ ===\n")

# Υπολογίζουμε τον συνολικό αριθμό γραμμών
total_rows = sum(1 for _ in open(file_path)) - 1  # -1 για header
print(f"Συνολικός αριθμός γραμμών: {total_rows:,}")

print("\n--- Ανάλυση κατανομής κλάσεων ---")
label_counts = {}
traffic_type_counts = {}
chunk_size = 50000

for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=['Label', 'Traffic Type']):
    # Label counts
    counts = chunk['Label'].value_counts()
    for label, count in counts.items():
        label_counts[label] = label_counts.get(label, 0) + count
    
    # Traffic Type counts
    type_counts = chunk['Traffic Type'].value_counts()
    for traffic_type, count in type_counts.items():
        traffic_type_counts[traffic_type] = traffic_type_counts.get(traffic_type, 0) + count

print("\nΚατανομή Labels:")
for label, count in sorted(label_counts.items()):
    print(f"{label}: {count:,} ({count/total_rows*100:.2f}%)")

print("\nΚατανομή Traffic Types:")
for traffic_type, count in sorted(traffic_type_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{traffic_type}: {count:,} ({count/total_rows*100:.2f}%)")

# ΜΕΘΟΔΟΣ 1: BALANCED ΣΤΡΩΜΑΤΟΠΟΙΗΜΕΝΗ ΔΕΙΓΜΑΤΟΛΗΨΙΑ
print("\n--- Μέθοδος 1: Balanced Στρωματοποιημένη Δειγματοληψία ---")

# Στόχος: Δημιουργία balanced dataset με καλή αναπαράσταση όλων των κλάσεων
target_samples_per_label = 25000  # Για κάθε label (Benign/Malicious)
target_samples_per_traffic_type = 5000  # Για κάθε traffic type

# Συλλέγουμε δείγματα ανά κλάση
benign_samples = []
malicious_samples = []
traffic_type_samples = {t_type: [] for t_type in traffic_type_counts.keys()}

print("\nΣυλλογή δειγμάτων...")
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    # Feature engineering
    chunk = create_combined_features(chunk)
    
    # Κρατάμε μόνο τις επιλεγμένες στήλες
    available_features = [col for col in all_features if col in chunk.columns]
    chunk = chunk[available_features]
    
    # Συλλέγουμε δείγματα ανά Label
    benign = chunk[chunk['Label'] == 'Benign']
    if len(benign) > 0:
        benign_samples.append(benign)
    
    malicious = chunk[chunk['Label'] == 'Malicious']
    if len(malicious) > 0:
        # Κρατάμε μικρό ποσοστό λόγω του μεγάλου όγκου
        sample_rate = min(0.003, target_samples_per_label / label_counts['Malicious'])
        n_samples = int(len(malicious) * sample_rate)
        if n_samples > 0:
            malicious_samples.append(malicious.sample(n=n_samples, random_state=42))
    
    # Συλλέγουμε δείγματα ανά Traffic Type
    for t_type in traffic_type_counts.keys():
        t_type_data = chunk[chunk['Traffic Type'] == t_type]
        if len(t_type_data) > 0:
            sample_rate = min(0.1, target_samples_per_traffic_type / traffic_type_counts[t_type])
            n_samples = max(1, int(len(t_type_data) * sample_rate))
            traffic_type_samples[t_type].append(t_type_data.sample(n=n_samples, random_state=42))

# Συνδυάζουμε τα δείγματα
print("\nΔημιουργία balanced dataset...")
all_benign = pd.concat(benign_samples, ignore_index=True) if benign_samples else pd.DataFrame()
all_malicious = pd.concat(malicious_samples, ignore_index=True) if malicious_samples else pd.DataFrame()

# Oversampling για Benign (αφού είναι πολύ λίγα)
if len(all_benign) > 0:
    benign_balanced = all_benign.sample(n=min(target_samples_per_label, len(all_benign)*10), 
                                       replace=True, random_state=42)
else:
    benign_balanced = pd.DataFrame()

# Sampling για Malicious
if len(all_malicious) > 0:
    malicious_balanced = all_malicious.sample(n=min(target_samples_per_label, len(all_malicious)), 
                                             replace=False, random_state=42)
else:
    malicious_balanced = pd.DataFrame()

# Συνδυασμός
dataset_sampled = pd.concat([benign_balanced, malicious_balanced], ignore_index=True)

# Shuffle
dataset_sampled = dataset_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nΤελικό μέγεθος balanced dataset: {len(dataset_sampled):,} γραμμές")
print(f"Label distribution: {dataset_sampled['Label'].value_counts().to_dict()}")
print(f"Traffic Type distribution: {dataset_sampled['Traffic Type'].value_counts().to_dict()}")

# Αποθήκευση
dataset_sampled.to_csv('../data/processed/dataset_1_balanced_sampled.csv', index=False)
print("Αποθηκεύτηκε ως: dataset_1_balanced_sampled.csv")

# ΜΕΘΟΔΟΣ 2: ΣΥΣΤΑΔΟΠΟΙΗΣΗ - MiniBatchKMeans με διατήρηση labels
print("\n--- Μέθοδος 2: Συσταδοποίηση με MiniBatchKMeans ---")

# Διαβάζουμε ένα μεγαλύτερο δείγμα για clustering
clustering_sample_size = 100000
print(f"Διάβασμα {clustering_sample_size:,} γραμμών για clustering...")

clustering_df = pd.read_csv(file_path, 
                           skiprows=lambda i: i > 0 and np.random.random() > clustering_sample_size/total_rows,
                           nrows=clustering_sample_size)

# Feature engineering
clustering_df = create_combined_features(clustering_df)

# Κρατάμε μόνο τις numeric στήλες για clustering
numeric_features = [col for col in all_features if col in clustering_df.columns 
                   and clustering_df[col].dtype in [np.float64, np.int64]
                   and col not in ['Label', 'Traffic Type', 'Traffic Subtype']]

X = clustering_df[numeric_features].fillna(0)
y_label = clustering_df['Label']
y_traffic = clustering_df['Traffic Type']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ξεχωριστό clustering για κάθε κλάση για να διατηρήσουμε την αντιπροσώπευση
print("\nClustering ανά κλάση...")
optimal_k_per_class = {'Benign': 50, 'Malicious': 150}  # Περισσότερα clusters για Malicious

cluster_representatives = []

for label in ['Benign', 'Malicious']:
    label_mask = y_label == label
    if label_mask.sum() == 0:
        continue
        
    X_label = X_scaled[label_mask]
    
    # Αν έχουμε λίγα δείγματα, κρατάμε όλα
    if len(X_label) < optimal_k_per_class[label]:
        representatives = clustering_df[label_mask].copy()
    else:
        # K-means clustering
        kmeans = MiniBatchKMeans(n_clusters=optimal_k_per_class[label], 
                                random_state=42, batch_size=1000)
        clusters = kmeans.fit_predict(X_label)
        
        # Για κάθε cluster, βρίσκουμε το πιο κοντινό σημείο στο center
        representatives = []
        for i in range(optimal_k_per_class[label]):
            cluster_mask = clusters == i
            if cluster_mask.sum() > 0:
                # Βρίσκουμε το σημείο πιο κοντά στο center
                cluster_center = kmeans.cluster_centers_[i]
                cluster_points = X_label[cluster_mask]
                distances = np.sum((cluster_points - cluster_center)**2, axis=1)
                closest_idx = np.argmin(distances)
                
                # Παίρνουμε το αντίστοιχο row από το original dataframe
                original_indices = np.where(label_mask)[0][cluster_mask]
                representative_idx = original_indices[closest_idx]
                representatives.append(clustering_df.iloc[representative_idx])
        
        representatives = pd.DataFrame(representatives)
    
    cluster_representatives.append(representatives)

# Συνδυασμός
dataset_kmeans = pd.concat(cluster_representatives, ignore_index=True)

print(f"\nΔημιουργήθηκε dataset με {len(dataset_kmeans)} representatives")
print(f"Label distribution: {dataset_kmeans['Label'].value_counts().to_dict()}")
print(f"Traffic Type distribution: {dataset_kmeans['Traffic Type'].value_counts().to_dict()}")

# Αποθήκευση
dataset_kmeans.to_csv('../data/processed/dataset_2_kmeans.csv', index=False)
print("Αποθηκεύτηκε ως: dataset_2_kmeans.csv")

# ΜΕΘΟΔΟΣ 3: ΣΥΣΤΑΔΟΠΟΙΗΣΗ - DBSCAN για ανίχνευση anomalies
print("\n--- Μέθοδος 3: Συσταδοποίηση με DBSCAN ---")

# Χρησιμοποιούμε μικρότερο δείγμα για DBSCAN
dbscan_sample_size = 50000
dbscan_df = clustering_df.sample(n=min(dbscan_sample_size, len(clustering_df)), random_state=42)
X_dbscan = dbscan_df[numeric_features].fillna(0)
X_dbscan_scaled = scaler.transform(X_dbscan)

# DBSCAN
print("Εκτέλεση DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=10, n_jobs=-1)
dbscan_clusters = dbscan.fit_predict(X_dbscan_scaled)

# Ανάλυση αποτελεσμάτων
n_clusters = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
n_noise = list(dbscan_clusters).count(-1)

print(f"\nΒρέθηκαν {n_clusters} clusters")
print(f"Noise points: {n_noise} ({n_noise/len(dbscan_clusters)*100:.2f}%)")

# Δημιουργία dataset με representatives από κάθε cluster + όλα τα noise points
dataset_dbscan_list = []

# Κρατάμε όλα τα noise points (πιθανές anomalies)
noise_mask = dbscan_clusters == -1
if noise_mask.sum() > 0:
    noise_data = dbscan_df[noise_mask]
    dataset_dbscan_list.append(noise_data)

# Για κάθε cluster, κρατάμε αντιπροσώπους
for cluster_id in set(dbscan_clusters):
    if cluster_id == -1:
        continue
        
    cluster_mask = dbscan_clusters == cluster_id
    cluster_data = dbscan_df[cluster_mask]
    
    # Κρατάμε 20% του cluster ή τουλάχιστον 10 σημεία
    n_representatives = max(10, int(len(cluster_data) * 0.2))
    n_representatives = min(n_representatives, len(cluster_data))
    
    # Stratified sampling βάσει Label και Traffic Type
    representatives = cluster_data.groupby(['Label', 'Traffic Type'], group_keys=False).apply(
        lambda x: x.sample(n=max(1, int(n_representatives * len(x) / len(cluster_data))), 
                          random_state=42)
    )
    
    dataset_dbscan_list.append(representatives)

dataset_dbscan = pd.concat(dataset_dbscan_list, ignore_index=True)

print(f"\nΔημιουργήθηκε dataset με {len(dataset_dbscan)} σημεία")
print(f"Label distribution: {dataset_dbscan['Label'].value_counts().to_dict()}")
print(f"Traffic Type distribution: {dataset_dbscan['Traffic Type'].value_counts().to_dict()}")

# Αποθήκευση
dataset_dbscan.to_csv('../data/processed/dataset_3_dbscan.csv', index=False)
print("Αποθηκεύτηκε ως: dataset_3_dbscan.csv")

# ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ
print("\n" + "="*60)
print("ΣΥΝΟΨΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ")
print("="*60)

print(f"\nΜείωση Διαστατικότητας:")
print(f"  - Αρχικές στήλες: {len(sample_df.columns)}")
print(f"  - Τελικές στήλες: {len(all_features)}")
print(f"  - Μείωση: {(1 - len(all_features)/len(sample_df.columns))*100:.1f}%")

print(f"\nΔημιουργήθηκαν 3 datasets:")
print(f"  1. Balanced Sampling: {len(dataset_sampled):,} rows")
print(f"     - Label balance: {dataset_sampled['Label'].value_counts().to_dict()}")
print(f"  2. K-Means Clustering: {len(dataset_kmeans):,} rows")
print(f"     - Preserves both classes: {dataset_kmeans['Label'].value_counts().to_dict()}")
print(f"  3. DBSCAN Clustering: {len(dataset_dbscan):,} rows")
print(f"     - Focus on anomalies: {n_noise} noise points")

# Visualization
print("\n=== ΟΠΤΙΚΟΠΟΙΗΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ===")

# Δημιουργία συγκριτικών γραφημάτων
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

datasets = {
    'Balanced Sampled': dataset_sampled,
    'K-Means': dataset_kmeans,
    'DBSCAN': dataset_dbscan
}

# Label distribution
for idx, (name, df) in enumerate(datasets.items()):
    ax = axes[0, idx]
    label_counts = df['Label'].value_counts()
    ax.bar(label_counts.index, label_counts.values)
    ax.set_title(f'{name} - Label Distribution')
    ax.set_ylabel('Count')
    
    # Προσθήκη ποσοστών
    total = label_counts.sum()
    for i, (label, count) in enumerate(label_counts.items()):
        ax.text(i, count + 50, f'{count/total*100:.1f}%', ha='center')

# Traffic Type distribution
for idx, (name, df) in enumerate(datasets.items()):
    ax = axes[1, idx]
    traffic_counts = df['Traffic Type'].value_counts().head(10)
    ax.bar(range(len(traffic_counts)), traffic_counts.values)
    ax.set_xticks(range(len(traffic_counts)))
    ax.set_xticklabels(traffic_counts.index, rotation=45, ha='right')
    ax.set_title(f'{name} - Top 10 Traffic Types')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('../reports/figures/dataset_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nΟλοκληρώθηκε η μείωση του dataset!")
print("Δημιουργήθηκαν τα αρχεία:")
print("  - dataset_1_balanced_sampled.csv")
print("  - dataset_2_kmeans.csv")
print("  - dataset_3_dbscan.csv")