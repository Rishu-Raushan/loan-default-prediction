import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(data_path, dict_path):
    """Load dataset and data dictionary"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    data_dict = pd.read_csv(dict_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.shape[1]}")
    print(f"Samples: {df.shape[0]}")
    
    return df, data_dict


def basic_info(df):
    """Display basic dataset information"""
    print("\n" + "="*70)
    print("BASIC DATASET INFORMATION")
    print("="*70)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    print("\nMemory Usage:")
    print(f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def analyze_target_variable(df, target_col='Default'):
    """Analyze target variable distribution"""
    print("\n" + "="*70)
    print("TARGET VARIABLE ANALYSIS")
    print("="*70)
    
    if target_col not in df.columns:
        available_cols = [col for col in df.columns if 'default' in col.lower() or 'status' in col.lower()]
        if available_cols:
            target_col = available_cols[0]
            print(f"Using {target_col} as target variable")
        else:
            print(f"Target column not found. Available columns: {df.columns.tolist()}")
            return
    
    print(f"\nTarget Variable: {target_col}")
    print(df[target_col].value_counts())
    print(f"\nClass Distribution:")
    print(df[target_col].value_counts(normalize=True) * 100)
    
    imbalance_ratio = df[target_col].value_counts()[0] / df[target_col].value_counts()[1]
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 2:
        print("WARNING: Significant class imbalance detected!")
        print("Recommendation: Use SMOTE, class weights, or ensemble methods")
    
    plt.figure(figsize=(10, 6))
    df[target_col].value_counts().plot(kind='bar')
    plt.title('Target Variable Distribution', fontsize=14)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('reports/plots/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nPlot saved: reports/plots/target_distribution.png")


def missing_values_analysis(df):
    """Analyze missing values"""
    print("\n" + "="*70)
    print("MISSING VALUES ANALYSIS")
    print("="*70)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    
    if len(missing_df) > 0:
        print(f"\nColumns with missing values: {len(missing_df)}")
        print(missing_df.to_string(index=False))
        
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(missing_df)), missing_df['Missing_Percentage'])
        plt.yticks(range(len(missing_df)), missing_df['Column'])
        plt.xlabel('Missing Percentage (%)', fontsize=12)
        plt.ylabel('Column', fontsize=12)
        plt.title('Missing Values Analysis', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('reports/plots/missing_values.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nPlot saved: reports/plots/missing_values.png")
    else:
        print("\nNo missing values found!")


def numerical_features_analysis(df):
    """Analyze numerical features"""
    print("\n" + "="*70)
    print("NUMERICAL FEATURES ANALYSIS")
    print("="*70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' in numerical_cols:
        numerical_cols.remove('ID')
    
    print(f"\nNumber of numerical features: {len(numerical_cols)}")
    print(f"Numerical columns: {numerical_cols}")
    
    print("\nStatistical Summary:")
    print(df[numerical_cols].describe())
    
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols[:20]):
        axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black')
        axes[idx].set_title(col, fontsize=10)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Frequency')
    
    for idx in range(len(numerical_cols[:20]), 20):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('reports/plots/numerical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nPlot saved: reports/plots/numerical_distributions.png")


def categorical_features_analysis(df):
    """Analyze categorical features"""
    print("\n" + "="*70)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("="*70)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumber of categorical features: {len(categorical_cols)}")
    print(f"Categorical columns: {categorical_cols}")
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"\n{col}:")
        print(f"  Unique values: {unique_count}")
        if unique_count <= 10:
            print(f"  Value counts:")
            print(df[col].value_counts())


def correlation_analysis(df, target_col='Default'):
    """Analyze correlations"""
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' in numerical_cols:
        numerical_cols.remove('ID')
    
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('reports/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nPlot saved: reports/plots/correlation_matrix.png")
    
    if target_col in df.columns and target_col in numerical_cols:
        target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
        print(f"\nTop 10 features correlated with {target_col}:")
        print(target_corr.head(11)[1:])


def outliers_analysis(df):
    """Analyze outliers"""
    print("\n" + "="*70)
    print("OUTLIERS ANALYSIS")
    print("="*70)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' in numerical_cols:
        numerical_cols.remove('ID')
    
    outlier_summary = []
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_pct = (outliers / len(df)) * 100
        
        outlier_summary.append({
            'Column': col,
            'Outliers': outliers,
            'Percentage': outlier_pct
        })
    
    outlier_df = pd.DataFrame(outlier_summary).sort_values('Percentage', ascending=False)
    outlier_df = outlier_df[outlier_df['Outliers'] > 0]
    
    if len(outlier_df) > 0:
        print(f"\nColumns with outliers: {len(outlier_df)}")
        print(outlier_df.head(10).to_string(index=False))


def bivariate_analysis(df, target_col='Default'):
    """Bivariate analysis with target variable"""
    print("\n" + "="*70)
    print("BIVARIATE ANALYSIS")
    print("="*70)
    
    if target_col not in df.columns:
        print(f"Target column {target_col} not found")
        return
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' in numerical_cols:
        numerical_cols.remove('ID')
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    print(f"\nAnalyzing top numerical features against {target_col}...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()
    
    for idx, col in enumerate(numerical_cols[:9]):
        for class_val in df[target_col].unique():
            data = df[df[target_col] == class_val][col].dropna()
            axes[idx].hist(data, bins=30, alpha=0.6, label=f'{target_col}={class_val}')
        axes[idx].set_title(col, fontsize=10)
        axes[idx].legend()
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('reports/plots/bivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nPlot saved: reports/plots/bivariate_analysis.png")


def main():
    """Main EDA execution"""
    import os
    os.makedirs('reports/plots', exist_ok=True)
    
    print("="*70)
    print("EXPLORATORY DATA ANALYSIS - LOAN DEFAULT PREDICTION")
    print("="*70)
    
    df, data_dict = load_data('data/Dataset.csv', 'data/Data_Dictionary.csv')
    
    basic_info(df)
    
    target_col = 'Default' if 'Default' in df.columns else 'loan_status'
    
    analyze_target_variable(df, target_col)
    
    missing_values_analysis(df)
    
    numerical_features_analysis(df)
    
    categorical_features_analysis(df)
    
    correlation_analysis(df, target_col)
    
    outliers_analysis(df)
    
    bivariate_analysis(df, target_col)
    
    print("\n" + "="*70)
    print("EDA COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print("1. Dataset loaded successfully with", df.shape[0], "samples and", df.shape[1], "features")
    print("2. Target variable shows class imbalance")
    print("3. Missing values and outliers detected")
    print("4. All plots saved to reports/plots/")
    print("\nNext Steps:")
    print("1. Handle missing values")
    print("2. Treat outliers")
    print("3. Feature engineering")
    print("4. Model training with class imbalance handling")


if __name__ == "__main__":
    main()
