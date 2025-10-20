import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Define the default probe IDs
aml_genes = ['M55150_at', 'X95735_at', 'Y12670_at', 'L08246_at', 
             'M62762_at', 'M63138_at', 'M57710_at', 'M81695_s_at']
all_genes = ['U22376_cds2_s_at', 'X59417_at', 'U05259_rna1_at', 'M31211_s_at', 
             'M91432_at', 'Z69881_at', 'D38073_at', 'U35451_at']

def load_data(script_dir):
    data_path = os.path.join(script_dir, '../data/raw/data_set_ALL_AML_train.csv')
    labels_path = os.path.join(script_dir, '../data/raw/actual.csv')

    try:
        train_data = pd.read_csv(data_path)
        print("Train data loaded successfully.")
        labels = pd.read_csv(labels_path)
        print("Labels data loaded successfully.")
        return train_data, labels
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

def get_user_input():
    num_genes = int(input("Enter the number of genes to test: "))
    use_default = input("Use default probe IDs? (yes/no): ").strip().lower()
    if use_default == 'yes':
        genes_to_test = aml_genes[:num_genes//2] + all_genes[:num_genes//2]
    else:
        print("Enter the probe IDs for the genes you want to test:")
        genes_to_test = [input(f"Gene {i+1}: ").strip() for i in range(num_genes)]
    return genes_to_test

def preprocess_train_data(train_data, genes_to_test):
    print("\n=== PREPROCESSING DATA ===")
    
    # Identify gene description and accession columns
    id_cols = ['Gene Description', 'Gene Accession Number']
    
    # Get all columns except the ID columns
    all_cols = [col for col in train_data.columns if col not in id_cols]
    
    # The data has pairs of columns: patient_number, call, patient_number, call, etc.
    # find all the patient number columns (they're numeric strings)
    patient_cols = []
    call_cols = []
    
    for i in range(0, len(all_cols), 2):
        if i + 1 < len(all_cols):
            # Check if this looks like a patient-call pair
            if all_cols[i].isdigit() and 'call' in all_cols[i + 1].lower():
                patient_cols.append(all_cols[i])
                call_cols.append(all_cols[i + 1])
    
    print(f"Found {len(patient_cols)} patient-call pairs")
    print(f"Patient columns: {patient_cols[:10]}...")
    print(f"Call columns: {call_cols[:10]}...")
    
    # Process each patient separately
    all_patient_data = []
    
    for i, patient_col in enumerate(patient_cols):
        call_col = call_cols[i]
        patient_id = int(patient_col)
        
        # Extract data for this patient
        patient_df = train_data[id_cols + [patient_col, call_col]].copy()
        patient_df = patient_df.rename(columns={
            patient_col: 'intensity',
            call_col: 'call'
        })
        patient_df['patient'] = patient_id
        
        all_patient_data.append(patient_df)
    
    # Combine all patient data
    train_long = pd.concat(all_patient_data, ignore_index=True)
    
    print(f"\nCombined data shape: {train_long.shape}")
    print(f"Sample call values: {train_long['call'].value_counts()}")
    
    # Convert intensity to numeric
    train_long['intensity'] = pd.to_numeric(train_long['intensity'], errors='coerce')
    
    # Filter for our genes of interest
    original_count = len(train_long)
    train_long = train_long[train_long['Gene Accession Number'].isin(genes_to_test)]
    print(f"Filtered from {original_count} to {len(train_long)} rows after gene selection")
    
    # Drop rows with NaN intensity
    train_long = train_long.dropna(subset=['intensity'])
    print(f"After dropping NaN intensities: {len(train_long)} rows")
    
    return train_long

def select_best_patients(merged_data, genes_to_test):
    """
    Select patients based on call quality (P > M > A) and ensure class balance
    """
    print(f"\n=== SELECTING PATIENTS ===")
    
    # see what call values we actually have
    print("Available call values in merged_data:")
    print(merged_data['call'].value_counts())
    
    # Create a patient-gene matrix with call quality
    patient_gene_matrix = merged_data.pivot_table(
        index=['patient', 'cancer'], 
        columns='Gene Accession Number', 
        values='call', 
        aggfunc='first'
    ).reset_index()
    
    print(f"Patient-gene matrix shape: {patient_gene_matrix.shape}")
    
    # Score call quality: P=2, M=1, A=0, (empty)=0
    call_scores = patient_gene_matrix.copy()
    for gene in genes_to_test:
        if gene in call_scores.columns:
            # Map call values to scores
            call_scores[gene] = call_scores[gene].map({'P': 2, 'M': 1, 'A': 0}).fillna(0)
        else:
            call_scores[gene] = 0
    
    # Calculate total score per patient
    call_scores['total_score'] = call_scores[genes_to_test].sum(axis=1)
    
    print(f"\nPatient call scores (top 10):")
    print(call_scores[['patient', 'cancer', 'total_score']].head(10))
    
    # Separate by cancer type
    aml_patients = call_scores[call_scores['cancer'] == 'AML']
    all_patients = call_scores[call_scores['cancer'] == 'ALL']
    
    print(f"\nAvailable AML patients: {len(aml_patients)}")
    print(f"Available ALL patients: {len(all_patients)}")
    
    # Sort patients by total score (descending) to prioritize better quality calls
    aml_patients = aml_patients.sort_values('total_score', ascending=False)
    all_patients = all_patients.sort_values('total_score', ascending=False)
    
    # Select top patients for each class
    aml_train_count = min(9, len(aml_patients))
    all_train_count = min(9, len(all_patients))
    
    aml_train_patients = aml_patients.head(aml_train_count)['patient'].tolist()
    all_train_patients = all_patients.head(all_train_count)['patient'].tolist()
    
    # For test set, take next best patients
    aml_test_patients = aml_patients.iloc[aml_train_count:aml_train_count+2]['patient'].tolist() if len(aml_patients) >= aml_train_count + 2 else []
    all_test_patients = all_patients.iloc[all_train_count:all_train_count+2]['patient'].tolist() if len(all_patients) >= all_train_count + 2 else []
    
    # If we don't have enough test patients, take from the end of train
    if len(aml_test_patients) < 2 and len(aml_train_patients) >= 11:
        aml_test_patients = aml_train_patients[-2:]
        aml_train_patients = aml_train_patients[:-2]
    
    if len(all_test_patients) < 2 and len(all_train_patients) >= 11:
        all_test_patients = all_train_patients[-2:]
        all_train_patients = all_train_patients[:-2]
    
    print(f"\nSelected {len(aml_train_patients)} AML train patients: {aml_train_patients}")
    print(f"Selected {len(all_train_patients)} ALL train patients: {all_train_patients}")
    print(f"Selected {len(aml_test_patients)} AML test patients: {aml_test_patients}")
    print(f"Selected {len(all_test_patients)} ALL test patients: {all_test_patients}")
    
    # Combine selected patients
    train_patients = aml_train_patients + all_train_patients
    test_patients = aml_test_patients + all_test_patients
    
    # Filter the original data
    train_data = merged_data[merged_data['patient'].isin(train_patients)]
    test_data = merged_data[merged_data['patient'].isin(test_patients)]
    
    return train_data, test_data, train_patients, test_patients

def merge_data(train_long, labels):
    # Convert patient in labels to integer
    labels['patient'] = labels['patient'].astype(int)
    
    # Merge the train data with the labels based on patient ID
    merged_data = pd.merge(train_long, labels, on='patient', how='inner')
    return merged_data

def generate_final_tables(train_data, test_data, genes_to_test):
    """Generate the final tables with intensity and call information"""
    
    # Combine train and test data
    combined_data = pd.concat([train_data, test_data])
    
    # Create intensity table
    intensity_table = combined_data.pivot_table(
        index=['patient', 'cancer'], 
        columns='Gene Accession Number', 
        values='intensity', 
        aggfunc='mean'
    ).reset_index()
    
    # Create call table
    call_table = combined_data.pivot_table(
        index=['patient', 'cancer'], 
        columns='Gene Accession Number', 
        values='call', 
        aggfunc='first'
    ).reset_index()
    
    # Ensure all genes are present in both tables
    for gene in genes_to_test:
        if gene not in intensity_table.columns:
            intensity_table[gene] = np.nan
        if gene not in call_table.columns:
            call_table[gene] = 'M'  # Default to Marginal if missing
    
    # Merge the tables
    final_table = pd.merge(intensity_table, call_table, on=['patient', 'cancer'], 
                          suffixes=('_intensity', '_call'))
    
    return final_table

def generate_patient_list_table(train_patients, test_patients, labels):
    """Generate a table listing all selected patients and their details"""
    # Get cancer types for each patient
    patient_info = labels[labels['patient'].isin(train_patients + test_patients)].copy()
    
    # Add dataset split information
    patient_info['split'] = 'train'
    patient_info.loc[patient_info['patient'].isin(test_patients), 'split'] = 'test'
    
    # Reorder columns
    patient_info = patient_info[['patient', 'cancer', 'split']].sort_values(['split', 'cancer', 'patient'])
    
    return patient_info

def get_call_quality_summary(train_long, selected_patients, genes_to_test):
    """Get call quality summary from the data"""
    # Filter for selected patients and genes
    call_data = train_long[
        (train_long['patient'].isin(selected_patients)) & 
        (train_long['Gene Accession Number'].isin(genes_to_test))
    ]
    
    call_counts = call_data['call'].value_counts()
    total_calls = len(call_data)
    
    print(f"\n=== CALL QUALITY SUMMARY ===")
    print(f"Total gene-patient observations: {total_calls}")
    if total_calls > 0:
        print(f"P (Present): {call_counts.get('P', 0)} ({call_counts.get('P', 0)/total_calls*100:.1f}%)")
        print(f"M (Marginal): {call_counts.get('M', 0)} ({call_counts.get('M', 0)/total_calls*100:.1f}%)")
        print(f"A (Absent): {call_counts.get('A', 0)} ({call_counts.get('A', 0)/total_calls*100:.1f}%)")
        
        # Also show per-patient call quality
        patient_call_quality = call_data.groupby('patient')['call'].apply(
            lambda x: f"P:{x.eq('P').sum()}, M:{x.eq('M').sum()}, A:{x.eq('A').sum()}"
        )
        print(f"\nCall Quality per Patient:")
        for patient, quality in patient_call_quality.items():
            print(f"Patient {patient}: {quality}")
    else:
        print("No call data found!")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_data, labels = load_data(script_dir)
    genes_to_test = get_user_input()
    
    print(f"\nTesting {len(genes_to_test)} genes: {genes_to_test}")
    
    # Preprocess data with the corrected structure
    train_long = preprocess_train_data(train_data, genes_to_test)
    merged_data = merge_data(train_long, labels)
    
    # Select patients based on call quality and class balance
    train_data, test_data, train_patients, test_patients = select_best_patients(merged_data, genes_to_test)
    all_selected_patients = train_patients + test_patients
    
    # Generate patient list table
    patient_list_table = generate_patient_list_table(train_patients, test_patients, labels)
    print(f"\n=== PATIENT SELECTION SUMMARY ===")
    print(f"Total patients selected: {len(all_selected_patients)}")
    print(f"Train patients: {len(train_patients)} (AML: {len([p for p in train_patients if labels[labels['patient'] == p]['cancer'].iloc[0] == 'AML'])}, ALL: {len([p for p in train_patients if labels[labels['patient'] == p]['cancer'].iloc[0] == 'ALL'])})")
    print(f"Test patients: {len(test_patients)} (AML: {len([p for p in test_patients if labels[labels['patient'] == p]['cancer'].iloc[0] == 'AML'])}, ALL: {len([p for p in test_patients if labels[labels['patient'] == p]['cancer'].iloc[0] == 'ALL'])})")
    
    # Generate final tables
    final_table = generate_final_tables(train_data, test_data, genes_to_test)
    
    # Save all tables
    final_table.to_csv('final_patient_gene_table.csv', index=False)
    patient_list_table.to_csv('selected_patients_list.csv', index=False)
    
    print(f"\n=== FILES SAVED ===")
    print(f"- final_patient_gene_table.csv: Contains gene intensities and calls for all selected patients")
    print(f"- selected_patients_list.csv: Contains the list of selected patients with cancer types and splits")
    
    # Print patient list
    print(f"\nSelected Patients:")
    print(patient_list_table.to_string(index=False))
    
    # Get call quality summary
    get_call_quality_summary(train_long, all_selected_patients, genes_to_test)

if __name__ == "__main__":
    main()