import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Ορίζουμε το path του αρχείου
file_path = '../data/raw/data.csv'

# 1. Πρώτη εξερεύνηση - Διαβάζουμε μόνο τις πρώτες γραμμές
print("=== ΒΑΣΙΚΕΣ ΠΛΗΡΟΦΟΡΙΕΣ ΓΙΑ ΤΟ DATASET ===\n")

# Διαβάζουμε μόνο τις πρώτες 1000 γραμμές για να δούμε τη δομή
sample_df = pd.read_csv(file_path, nrows=1000)

print(f"Στήλες του dataset: {sample_df.shape[1]}")
print(f"Ονόματα στηλών:\n{list(sample_df.columns)}\n")

# Εμφανίζουμε τους τύπους δεδομένων
print("Τύποι δεδομένων ανά στήλη:")
print(sample_df.dtypes)

# 2. Υπολογισμός βασικών στατιστικών με chunking
print("\n=== ΥΠΟΛΟΓΙΣΜΟΣ ΒΑΣΙΚΩΝ ΣΤΑΤΙΣΤΙΚΩΝ (με chunking) ===\n")

# Αρχικοποίηση μεταβλητών για στατιστικά
chunk_size = 50000  # Μέγεθος κάθε chunk
n_rows = 0
numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()

# Dictionaries για αποθήκευση στατιστικών
sums = defaultdict(float)
sums_squared = defaultdict(float)
mins = defaultdict(lambda: float('inf'))
maxs = defaultdict(lambda: float('-inf'))
counts = defaultdict(int)

# Για κατηγορικές μεταβλητές
categorical_cols = sample_df.select_dtypes(include=['object']).columns.tolist()
unique_values = defaultdict(set)

print("Επεξεργασία chunks...")
chunk_counter = 0

# Διάβασμα και επεξεργασία κάθε chunk
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    chunk_counter += 1
    n_rows += len(chunk)
    
    # Υπολογισμοί για αριθμητικές στήλες
    for col in numeric_cols:
        if col in chunk.columns:
            # Αφαιρούμε NaN τιμές
            valid_data = chunk[col].dropna()
            
            if len(valid_data) > 0:
                sums[col] += valid_data.sum()
                sums_squared[col] += (valid_data ** 2).sum()
                mins[col] = min(mins[col], valid_data.min())
                maxs[col] = max(maxs[col], valid_data.max())
                counts[col] += len(valid_data)
    
    # Για κατηγορικές μεταβλητές
    for col in categorical_cols:
        if col in chunk.columns:
            unique_values[col].update(chunk[col].dropna().unique())
    
    # Εμφάνιση προόδου
    if chunk_counter % 10 == 0:
        print(f"Επεξεργασία chunk {chunk_counter}, Σύνολο γραμμών: {n_rows:,}")

print(f"\nΣυνολικός αριθμός γραμμών: {n_rows:,}")

# Υπολογισμός τελικών στατιστικών
statistics = pd.DataFrame()

for col in numeric_cols:
    if counts[col] > 0:
        mean = sums[col] / counts[col]
        variance = (sums_squared[col] / counts[col]) - (mean ** 2)
        std = np.sqrt(variance) if variance > 0 else 0
        
        statistics[col] = {
            'count': counts[col],
            'mean': mean,
            'std': std,
            'min': mins[col],
            'max': maxs[col]
        }

print("\nΒασικά στατιστικά για αριθμητικές στήλες:")
print(statistics.T.head(10))  # Εμφανίζουμε τις πρώτες 10 στήλες

# 3. Ανάλυση κατηγορικών μεταβλητών
print("\n=== ΑΝΑΛΥΣΗ ΚΑΤΗΓΟΡΙΚΩΝ ΜΕΤΑΒΛΗΤΩΝ ===\n")

for col in categorical_cols[:5]:  # Εμφανίζουμε τις πρώτες 5 κατηγορικές
    print(f"\n{col}:")
    print(f"Αριθμός μοναδικών τιμών: {len(unique_values[col])}")
    if len(unique_values[col]) <= 10:
        print(f"Τιμές: {unique_values[col]}")

# 4. Ανάλυση κατανομών - Δειγματοληψία για γραφήματα
print("\n=== ΔΗΜΙΟΥΡΓΙΑ ΓΡΑΦΗΜΑΤΩΝ (με δειγματοληψία) ===\n")

# Διαβάζουμε ένα τυχαίο δείγμα για τα γραφήματα
sample_size = 10000
sample_df = pd.read_csv(file_path, 
                        skiprows=lambda i: i > 0 and np.random.random() > sample_size/n_rows,
                        nrows=sample_size)

print(f"Μέγεθος δείγματος για γραφήματα: {len(sample_df)}")

# Δημιουργία γραφημάτων για τις πιο σημαντικές στήλες
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Κατανομές Επιλεγμένων Χαρακτηριστικών', fontsize=16)

# Ρυθμίσεις απόστασης
plt.subplots_adjust(wspace=0.14, hspace=0.315, top=0.91, bottom=0.075)

# Επιλογή στηλών για visualization
important_cols = ['Total Fwd Packet', 'Total Bwd packets', 'Flow Duration', 'Flow Bytes/s']
available_cols = [col for col in important_cols if col in sample_df.columns]

for idx, col in enumerate(available_cols[:4]):
    ax = axes[idx//2, idx%2]
    
    # Αφαιρούμε outliers για καλύτερη οπτικοποίηση
    data = sample_df[col].dropna()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    ax.hist(filtered_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title(f'Κατανομή: {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Συχνότητα')
    ax.grid(True, alpha=0.3)

plt.savefig('../data/reports/figures/distributions.png', dpi=300, bbox_inches='tight')
plt.show()


# 5. Ανάλυση ετικετών (Labels)
print("\n=== ΑΝΑΛΥΣΗ ΕΤΙΚΕΤΩΝ ===\n")

label_counts = defaultdict(int)
traffic_type_counts = defaultdict(int)
traffic_subtype_counts = defaultdict(int)

# Μετρήσεις με chunking
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    if 'Label' in chunk.columns:
        label_counts_chunk = chunk['Label'].value_counts()
        for label, count in label_counts_chunk.items():
            label_counts[label] += count
    
    if 'Traffic Type' in chunk.columns:
        type_counts_chunk = chunk['Traffic Type'].value_counts()
        for t_type, count in type_counts_chunk.items():
            traffic_type_counts[t_type] += count
    
    if 'Traffic Subtype' in chunk.columns:
        subtype_counts_chunk = chunk['Traffic Subtype'].value_counts()
        for subtype, count in subtype_counts_chunk.items():
            traffic_subtype_counts[subtype] += count

# Εμφάνιση αποτελεσμάτων
print("Κατανομή Labels:")
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{label}: {count:,} ({count/n_rows*100:.2f}%)")

if traffic_type_counts:
    print("\nΚατανομή Traffic Types:")
    for t_type, count in sorted(traffic_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{t_type}: {count:,} ({count/n_rows*100:.2f}%)")

# 6. Συσχετίσεις - Υπολογισμός με δειγματοληψία
print("\n=== ΑΝΑΛΥΣΗ ΣΥΣΧΕΤΙΣΕΩΝ ===\n")

# Χρησιμοποιούμε μεγαλύτερο δείγμα για τις συσχετίσεις
correlation_sample = pd.read_csv(file_path, nrows=50000)

# Επιλέγουμε σημαντικές αριθμητικές στήλες
important_numeric_cols = [
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Flow Bytes/s', 'Flow Packets/s', 'Average Packet Size',
    'Fwd Packets/s', 'Bwd Packets/s'
]

available_numeric_cols = [col for col in important_numeric_cols if col in correlation_sample.columns]
correlation_matrix = correlation_sample[available_numeric_cols].corr()

# Heatmap συσχετίσεων
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Πίνακας Συσχετίσεων Βασικών Χαρακτηριστικών', fontsize=14)
plt.subplots_adjust(bottom=0.31, right=0.78, top=0.95)
plt.savefig('../data/reports/figures/correlations.png', dpi=300, bbox_inches='tight')
plt.show()


# 7. Ανάλυση Protocols
print("\n=== ΑΝΑΛΥΣΗ ΠΡΩΤΟΚΟΛΛΩΝ ===\n")

protocol_counts = defaultdict(int)
protocol_mapping = {
    6: 'TCP',
    17: 'UDP',
    1: 'ICMP',
    0: 'HOPOPT'
}

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    if 'Protocol' in chunk.columns:
        protocol_counts_chunk = chunk['Protocol'].value_counts()
        for protocol, count in protocol_counts_chunk.items():
            protocol_counts[protocol] += count

print("Κατανομή Πρωτοκόλλων:")
for protocol, count in sorted(protocol_counts.items(), key=lambda x: x[1], reverse=True):
    protocol_name = protocol_mapping.get(protocol, f'Unknown ({protocol})')
    print(f"{protocol_name}: {count:,} ({count/n_rows*100:.3f}%)")

# 8. Ανίχνευση μοτίβων
print("\n=== ΑΝΑΛΥΣΗ ΜΟΤΙΒΩΝ ===\n")

# Ανάλυση χρονικών μοτίβων αν υπάρχει timestamp
if 'Timestamp' in sample_df.columns:
    print("Ανάλυση χρονικών μοτίβων...")
    # Εδώ μπορείτε να προσθέσετε ανάλυση timestamps
    
# Ανάλυση μεγέθους πακέτων
print("\nΜοτίβα μεγέθους πακέτων:")
print(f"Μέσο μέγεθος Forward πακέτων: {statistics['Fwd Packet Length Mean']['mean']:.2f}")
print(f"Μέσο μέγεθος Backward πακέτων: {statistics['Bwd Packet Length Mean']['mean']:.2f}")

# Σύνοψη ευρημάτων
print("\n=== ΣΥΝΟΨΗ ΕΥΡΗΜΑΤΩΝ ===\n")
print("1. Το dataset περιέχει δεδομένα κίνησης δικτύου με διάφορα χαρακτηριστικά ροών")
print("2. Υπάρχουν τόσο καλοήθεις (Benign) όσο και κακόβουλες κινήσεις")
print("3. Τα κύρια πρωτόκολλα είναι TCP και UDP")
print("4. Παρατηρούνται συσχετίσεις μεταξύ του μεγέθους πακέτων και της διάρκειας ροής")
print("5. Η κατανομή πολλών χαρακτηριστικών είναι ασύμμετρη με πολλές ακραίες τιμές")