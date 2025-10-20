import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# Create directories for saving plots
os.makedirs('plots', exist_ok=True)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the generated CSV files"""
    try:
        final_table = pd.read_csv('final_patient_gene_table.csv')
        patient_list = pd.read_csv('selected_patients_list.csv')
        print(f"Final table loaded: {final_table.shape[0]} patients, {final_table.shape[1]} columns")
        print(f"Patient list loaded: {len(patient_list)} patients")
        return final_table, patient_list
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please run preprocessing.py first to generate the CSV files")
        return None, None

def plot_patient_distribution(patient_list):
    """Plot distribution of patients by cancer type and split"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Cancer type distribution
    cancer_counts = patient_list['cancer'].value_counts()
    axes[0].pie(cancer_counts.values, labels=cancer_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=['lightcoral', 'lightblue'])
    axes[0].set_title('Distribution of Cancer Types\n(All Patients)')
    
    # Plot 2: Split distribution by cancer type
    split_counts = pd.crosstab(patient_list['cancer'], patient_list['split'])
    split_counts.plot(kind='bar', ax=axes[1], color=['lightgreen', 'orange'])
    axes[1].set_title('Patient Distribution by Cancer Type and Split')
    axes[1].set_xlabel('Cancer Type')
    axes[1].set_ylabel('Number of Patients')
    axes[1].legend(title='Split')
    axes[1].tick_params(axis='x', rotation=0)
    
    # Add count labels on bars
    for p in axes[1].patches:
        axes[1].annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/patient_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_gene_expression_heatmap(final_table):
    """Create a heatmap of gene expression intensities"""
    # Extract intensity columns
    intensity_cols = [col for col in final_table.columns if 'intensity' in col and 'call' not in col]
    
    if not intensity_cols:
        print("No intensity columns found")
        return
    
    # Create a subset for heatmap
    heatmap_data = final_table[intensity_cols]
    
    # Get gene names (remove '_intensity' suffix)
    gene_names = [col.replace('_intensity', '') for col in intensity_cols]
    heatmap_data.columns = gene_names
    
    # Add patient and cancer info as index
    patient_labels = final_table['patient'].astype(str) + '_' + final_table['cancer'] + '_' + final_table.get('split', 'unknown')
    heatmap_data.index = patient_labels
    
    # Standardize the data for better visualization
    scaler = StandardScaler()
    heatmap_data_standardized = pd.DataFrame(
        scaler.fit_transform(heatmap_data),
        columns=heatmap_data.columns,
        index=heatmap_data.index
    )
    
    # Create the heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data_standardized, 
                cmap='RdBu_r', 
                center=0,
                cbar_kws={'label': 'Standardized Intensity'},
                yticklabels=True)
    plt.title('Gene Expression Heatmap\n(Standardized Intensity Values)\nColor: Red=High, Blue=Low')
    plt.xlabel('Genes')
    plt.ylabel('Patients (PatientID_CancerType_Split)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/gene_expression_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_call_quality_heatmap(final_table):
    """Create a heatmap of call quality (P/M/A)"""
    # Extract call columns
    call_cols = [col for col in final_table.columns if '_call' in col]
    
    if not call_cols:
        print("No call columns found")
        return
    
    # Create a subset for call heatmap
    call_data = final_table[call_cols]
    
    # Get gene names (remove '_call' suffix)
    gene_names = [col.replace('_call', '') for col in call_cols]
    call_data.columns = gene_names
    
    # Add patient and cancer info as index
    patient_labels = final_table['patient'].astype(str) + '_' + final_table['cancer'] + '_' + final_table.get('split', 'unknown')
    call_data.index = patient_labels
    
    # Convert calls to numerical values for coloring
    call_numeric = call_data.replace({'P': 2, 'M': 1, 'A': 0})
    
    # Create the heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(call_numeric, 
                cmap=['red', 'yellow', 'green'],  # A=red, M=yellow, P=green
                cbar_kws={'label': 'Call Quality', 'ticks': [0, 1, 2]},
                yticklabels=True,
                vmin=0, vmax=2)
    
    # Customize colorbar labels
    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticklabels(['A (Absent)', 'M (Marginal)', 'P (Present)'])
    
    plt.title('Gene Call Quality Heatmap\nGreen=Present, Yellow=Marginal, Red=Absent')
    plt.xlabel('Genes')
    plt.ylabel('Patients (PatientID_CancerType_Split)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/call_quality_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_expression_by_cancer(final_table):
    """Plot gene expression patterns by cancer type"""
    # Extract intensity columns
    intensity_cols = [col for col in final_table.columns if 'intensity' in col and 'call' not in col]
    
    if not intensity_cols:
        print("No intensity columns found")
        return
    
    # Prepare data for plotting
    plot_data = final_table[['cancer'] + intensity_cols].melt(
        id_vars=['cancer'], 
        var_name='gene', 
        value_name='intensity'
    )
    
    # Clean gene names
    plot_data['gene'] = plot_data['gene'].str.replace('_intensity', '')
    
    # Create boxplot
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=plot_data, x='gene', y='intensity', hue='cancer')
    plt.title('Gene Expression Intensity by Cancer Type\n(Boxplot Distribution)')
    plt.xlabel('Genes')
    plt.ylabel('Expression Intensity')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Cancer Type', loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/expression_by_cancer.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_analysis(final_table):
    """Plot correlation matrix between genes"""
    # Extract intensity columns
    intensity_cols = [col for col in final_table.columns if 'intensity' in col and 'call' not in col]
    
    if not intensity_cols:
        print("No intensity columns found")
        return
    
    # Calculate correlation matrix
    corr_matrix = final_table[intensity_cols].corr()
    
    # Clean gene names
    corr_matrix.columns = [col.replace('_intensity', '') for col in corr_matrix.columns]
    corr_matrix.index = [col.replace('_intensity', '') for col in corr_matrix.index]
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, 
                mask=mask,
                cmap='coolwarm', 
                center=0,
                annot=True, 
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'},
                annot_kws={'size': 8})
    plt.title('Gene Expression Correlation Matrix\n(Pearson Correlation)')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca_analysis(final_table):
    """Perform PCA and plot the results"""
    # Extract intensity columns
    intensity_cols = [col for col in final_table.columns if 'intensity' in col and 'call' not in col]
    
    if len(intensity_cols) < 2:
        print("Not enough genes for PCA")
        return
    
    # Prepare data for PCA
    pca_data = final_table[intensity_cols]
    
    # Standardize the data
    scaler = StandardScaler()
    pca_data_standardized = scaler.fit_transform(pca_data)
    
    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(pca_data_standardized)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame(data=principal_components, 
                         columns=['PC1', 'PC2'])
    pca_df['cancer'] = final_table['cancer'].values
    pca_df['patient'] = final_table['patient'].astype(str)
    pca_df['split'] = final_table.get('split', 'unknown')
    
    # Plot PCA results
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with different markers for train/test
    markers = {'train': 'o', 'test': 's'}
    colors = {'AML': 'red', 'ALL': 'blue'}
    
    for split in pca_df['split'].unique():
        for cancer in pca_df['cancer'].unique():
            subset = pca_df[(pca_df['split'] == split) & (pca_df['cancer'] == cancer)]
            if len(subset) > 0:
                plt.scatter(subset['PC1'], subset['PC2'], 
                           c=colors[cancer],
                           marker=markers.get(split, 'o'),
                           s=100, alpha=0.7,
                           label=f'{cancer} ({split})')
    
    # Add patient labels
    for i, row in pca_df.iterrows():
        plt.annotate(row['patient'], (row['PC1'], row['PC2']), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA: Gene Expression Patterns\n(Colored by Cancer Type, Markers by Split)')
    plt.legend(title='Cancer & Split')
    plt.grid(True, alpha=0.3)
    
    # Add explained variance info
    plt.text(0.02, 0.98, f'Total Variance Explained: {pca.explained_variance_ratio_.sum():.2%}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print explained variance
    print(f"PCA Explained Variance: PC1: {pca.explained_variance_ratio_[0]:.2%}, "
          f"PC2: {pca.explained_variance_ratio_[1]:.2%}")

def plot_patient_clustering(final_table):
    """Perform and visualize hierarchical clustering of patients"""
    # Extract intensity columns
    intensity_cols = [col for col in final_table.columns if 'intensity' in col and 'call' not in col]
    
    if not intensity_cols:
        print("No intensity columns found")
        return
    
    # Prepare data for clustering
    cluster_data = final_table[intensity_cols]
    
    # Standardize the data
    scaler = StandardScaler()
    cluster_data_standardized = scaler.fit_transform(cluster_data)
    
    # Perform hierarchical clustering
    Z = linkage(cluster_data_standardized, method='ward')
    
    # Create labels with patient ID, cancer type and split
    labels = [f"P{pid}_{cancer}_{split}" for pid, cancer, split in 
              zip(final_table['patient'], final_table['cancer'], final_table.get('split', 'unknown'))]
    
    # Create dendrogram
    plt.figure(figsize=(14, 8))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8)
    plt.title('Hierarchical Clustering of Patients\n(Based on Standardized Gene Expression)')
    plt.xlabel('Patients (PatientID_CancerType_Split)')
    plt.ylabel('Distance (Ward)')
    plt.tight_layout()
    plt.savefig('plots/patient_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_summary_statistics(final_table):
    """Plot summary statistics for the dataset"""
    # Extract intensity columns
    intensity_cols = [col for col in final_table.columns if 'intensity' in col and 'call' not in col]
    
    if not intensity_cols:
        print("No intensity columns found")
        return
    
    # Calculate summary statistics
    summary_stats = final_table[intensity_cols].describe()
    
    # Clean gene names
    summary_stats.columns = [col.replace('_intensity', '') for col in summary_stats.columns]
    
    # Plot summary statistics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mean expression
    summary_stats.loc['mean'].plot(kind='bar', ax=axes[0,0], color='skyblue', alpha=0.7)
    axes[0,0].set_title('Mean Gene Expression Intensity')
    axes[0,0].set_ylabel('Intensity')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # Standard deviation
    summary_stats.loc['std'].plot(kind='bar', ax=axes[0,1], color='lightcoral', alpha=0.7)
    axes[0,1].set_title('Standard Deviation of Gene Expression')
    axes[0,1].set_ylabel('Standard Deviation')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Min expression
    summary_stats.loc['min'].plot(kind='bar', ax=axes[1,0], color='lightgreen', alpha=0.7)
    axes[1,0].set_title('Minimum Gene Expression')
    axes[1,0].set_ylabel('Minimum Intensity')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Max expression
    summary_stats.loc['max'].plot(kind='bar', ax=axes[1,1], color='gold', alpha=0.7)
    axes[1,1].set_title('Maximum Gene Expression')
    axes[1,1].set_ylabel('Maximum Intensity')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_call_quality_summary(final_table):
    """Plot summary of call quality across genes and patients"""
    # Extract call columns
    call_cols = [col for col in final_table.columns if '_call' in col]
    
    if not call_cols:
        print("No call columns found")
        return
    
    # Calculate call statistics per gene
    call_stats = {}
    for col in call_cols:
        gene = col.replace('_call', '')
        calls = final_table[col].value_counts()
        call_stats[gene] = {
            'P': calls.get('P', 0),
            'M': calls.get('M', 0),
            'A': calls.get('A', 0),
            'Total': len(final_table)
        }
    
    call_df = pd.DataFrame(call_stats).T
    call_df['P_percent'] = (call_df['P'] / call_df['Total']) * 100
    call_df['M_percent'] = (call_df['M'] / call_df['Total']) * 100
    call_df['A_percent'] = (call_df['A'] / call_df['Total']) * 100
    
    # Plot call quality by gene
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Stacked bar chart
    call_df[['P_percent', 'M_percent', 'A_percent']].plot(kind='bar', stacked=True, 
                                                         ax=ax1, 
                                                         color=['green', 'yellow', 'red'])
    ax1.set_title('Call Quality Distribution by Gene\n(Percentage)')
    ax1.set_ylabel('Percentage of Patients')
    ax1.set_xlabel('Genes')
    ax1.legend(['Present (P)', 'Marginal (M)', 'Absent (A)'])
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Total counts
    call_df[['P', 'M', 'A']].plot(kind='bar', ax=ax2, 
                                 color=['green', 'yellow', 'red'], alpha=0.7)
    ax2.set_title('Call Quality Counts by Gene')
    ax2.set_ylabel('Number of Patients')
    ax2.set_xlabel('Genes')
    ax2.legend(['Present (P)', 'Marginal (M)', 'Absent (A)'])
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/call_quality_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all visualizations"""
    print("Loading and visualizing preprocessed data...")
    
    # Load data
    final_table, patient_list = load_data()
    if final_table is None or patient_list is None:
        return
    
    # Add split information to final_table if not present
    if 'split' not in final_table.columns:
        split_mapping = patient_list.set_index('patient')['split'].to_dict()
        final_table['split'] = final_table['patient'].map(split_mapping)
    
    # Display basic information
    print("\n=== DATASET INFO ===")
    print(f"Final table shape: {final_table.shape}")
    print(f"Patients: {len(final_table)}")
    print(f"Genes: {len([col for col in final_table.columns if 'intensity' in col and 'call' not in col])}")
    print(f"Cancer distribution:")
    print(final_table['cancer'].value_counts())
    print(f"Split distribution:")
    print(final_table['split'].value_counts())
    
    # Run visualizations
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # 1. Patient distribution
    print("1. Plotting patient distribution...")
    plot_patient_distribution(patient_list)
    
    # 2. Summary statistics
    print("2. Plotting summary statistics...")
    plot_summary_statistics(final_table)
    
    # 3. Gene expression by cancer type
    print("3. Plotting gene expression by cancer type...")
    plot_expression_by_cancer(final_table)
    
    # 4. Correlation heatmap
    print("4. Plotting gene correlation heatmap...")
    plot_correlation_analysis(final_table)
    
    # 5. Expression heatmap
    print("5. Plotting expression heatmap...")
    plot_gene_expression_heatmap(final_table)
    
    # 6. Call quality heatmap
    print("6. Plotting call quality heatmap...")
    plot_call_quality_heatmap(final_table)
    
    # 7. Call quality summary
    print("7. Plotting call quality summary...")
    plot_call_quality_summary(final_table)
    
    # 8. PCA analysis
    print("8. Performing PCA analysis...")
    plot_pca_analysis(final_table)
    
    # 9. Patient clustering
    print("9. Performing patient clustering...")
    plot_patient_clustering(final_table)
    
    print("\n=== ALL VISUALIZATIONS COMPLETED ===")
    print("All plots have been saved to the 'plots' directory.")
    
    # Print some additional statistics
    print("\n=== ADDITIONAL STATISTICS ===")
    intensity_cols = [col for col in final_table.columns if 'intensity' in col and 'call' not in col]
    print(f"Average intensity range across genes: {final_table[intensity_cols].mean().min():.2f} to {final_table[intensity_cols].mean().max():.2f}")
    print(f"Most variable gene: {final_table[intensity_cols].std().idxmax().replace('_intensity', '')}")
    print(f"Least variable gene: {final_table[intensity_cols].std().idxmin().replace('_intensity', '')}")

if __name__ == "__main__":
    main()    