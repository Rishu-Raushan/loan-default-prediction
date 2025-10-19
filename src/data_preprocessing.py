import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for loan default prediction.
    Handles missing values, outliers, scaling, and encoding.
    """
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_stats = {}
        
    def load_data(self, filepath):
        """Load dataset from CSV file"""
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def analyze_missing_values(self, df):
        """Analyze and report missing values"""
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
        
        print("\nMissing Values Analysis:")
        print(missing_df.to_string(index=False))
        return missing_df
    
    def handle_missing_values(self, df):
        """Handle missing values with sophisticated strategies"""
        print("\nHandling missing values...")
        df_clean = df.copy()
        
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        if 'ID' in numerical_cols:
            numerical_cols.remove('ID')
        if 'Default' in numerical_cols:
            numerical_cols.remove('Default')
        if 'loan_status' in numerical_cols:
            numerical_cols.remove('loan_status')
            
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100
                
                if missing_pct > self.config['preprocessing']['missing_value_threshold'] * 100:
                    print(f"Dropping column {col} due to {missing_pct:.2f}% missing values")
                    df_clean = df_clean.drop(columns=[col])
                else:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
                    self.imputers[col] = median_val
                    print(f"Imputed {col} with median: {median_val:.2f}")
        
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                self.imputers[col] = mode_val
                print(f"Imputed {col} with mode: {mode_val}")
        
        return df_clean
    
    def detect_outliers(self, df, columns=None):
        """Detect outliers using IQR method"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'ID' in columns:
                columns.remove('ID')
            if 'Default' in columns:
                columns.remove('Default')
            if 'loan_status' in columns:
                columns.remove('loan_status')
        
        outlier_summary = {}
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.config['preprocessing']['outlier_threshold'] * IQR
            upper_bound = Q3 + self.config['preprocessing']['outlier_threshold'] * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(df)) * 100
            
            outlier_summary[col] = {
                'count': outliers,
                'percentage': outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return outlier_summary
    
    def handle_outliers(self, df):
        """Handle outliers using capping method"""
        print("\nHandling outliers...")
        df_clean = df.copy()
        
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if 'ID' in numerical_cols:
            numerical_cols.remove('ID')
        if 'Default' in numerical_cols:
            numerical_cols.remove('Default')
        if 'loan_status' in numerical_cols:
            numerical_cols.remove('loan_status')
        
        outlier_info = self.detect_outliers(df_clean, numerical_cols)
        
        for col in numerical_cols:
            if outlier_info[col]['count'] > 0:
                lower = outlier_info[col]['lower_bound']
                upper = outlier_info[col]['upper_bound']
                
                df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])
                
                print(f"Capped {col}: {outlier_info[col]['count']} outliers ({outlier_info[col]['percentage']:.2f}%)")
                
                self.feature_stats[col] = {
                    'lower_bound': lower,
                    'upper_bound': upper
                }
        
        return df_clean
    
    def encode_categorical_features(self, df, target_col=None, fit=True):
        """Encode categorical features"""
        print("\nEncoding categorical features...")
        df_encoded = df.copy()
        
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            unique_values = df_encoded[col].nunique()
            
            if unique_values == 2:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = {'type': 'label', 'encoder': le}
                else:
                    if col in self.encoders and self.encoders[col]['type'] == 'label':
                        # Handle unknown categories
                        known_classes = set(self.encoders[col]['encoder'].classes_)
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: x if x in known_classes else self.encoders[col]['encoder'].classes_[0]
                        )
                        df_encoded[col] = self.encoders[col]['encoder'].transform(df_encoded[col].astype(str))
                print(f"Label encoded {col} (binary)")
                
            elif unique_values <= 10:
                if fit:
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    self.encoders[col] = {'type': 'onehot', 'columns': list(dummies.columns)}
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                else:
                    if col in self.encoders and self.encoders[col]['type'] == 'onehot':
                        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                        for dummy_col in self.encoders[col]['columns']:
                            if dummy_col not in dummies.columns:
                                df_encoded[dummy_col] = 0
                        # Only include columns that were in training
                        existing_cols = [c for c in self.encoders[col]['columns'] if c in dummies.columns]
                        df_encoded = pd.concat([df_encoded, dummies[existing_cols]], axis=1)
                        # Add missing columns with zeros
                        for dummy_col in self.encoders[col]['columns']:
                            if dummy_col not in existing_cols:
                                df_encoded[dummy_col] = 0
                df_encoded = df_encoded.drop(columns=[col])
                print(f"One-hot encoded {col} ({unique_values} categories)")
                
            else:
                if target_col is not None and fit:
                    target_mean = df_encoded.groupby(col)[target_col].mean()
                    df_encoded[col] = df_encoded[col].map(target_mean)
                    self.encoders[col] = {'type': 'target', 'mapping': target_mean.to_dict()}
                    print(f"Target encoded {col} ({unique_values} categories)")
                elif col in self.encoders and not fit:
                    if self.encoders[col]['type'] == 'target':
                        df_encoded[col] = df_encoded[col].map(self.encoders[col]['mapping'])
                        df_encoded[col].fillna(df_encoded[col].mean(), inplace=True)
                    elif self.encoders[col]['type'] == 'label':
                        # Handle unknown categories
                        known_classes = set(self.encoders[col]['encoder'].classes_)
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: x if x in known_classes else self.encoders[col]['encoder'].classes_[0]
                        )
                        df_encoded[col] = self.encoders[col]['encoder'].transform(df_encoded[col].astype(str))
                else:
                    le = LabelEncoder()
                    if fit:
                        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                        self.encoders[col] = {'type': 'label', 'encoder': le}
                    else:
                        if col in self.encoders and self.encoders[col]['type'] == 'label':
                            # Handle unknown categories
                            known_classes = set(self.encoders[col]['encoder'].classes_)
                            df_encoded[col] = df_encoded[col].apply(
                                lambda x: x if x in known_classes else self.encoders[col]['encoder'].classes_[0]
                            )
                            df_encoded[col] = self.encoders[col]['encoder'].transform(df_encoded[col].astype(str))
                    print(f"Label encoded {col} ({unique_values} categories)")
        
        return df_encoded
    
    def scale_features(self, df, columns=None, fit=True):
        """Scale numerical features"""
        print("\nScaling features...")
        df_scaled = df.copy()
        
        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            if 'ID' in columns:
                columns.remove('ID')
            if 'Default' in columns:
                columns.remove('Default')
            if 'loan_status' in columns:
                columns.remove('loan_status')
        
        if fit:
            scaler = StandardScaler()
            df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
            self.scalers['standard'] = scaler
            self.scalers['columns'] = columns  # Store which columns were scaled
            print(f"Scaled {len(columns)} features using StandardScaler")
        else:
            if 'standard' in self.scalers and 'columns' in self.scalers:
                # Only scale columns that exist in both train and test
                scale_cols = [col for col in self.scalers['columns'] if col in df_scaled.columns]
                if scale_cols:
                    # Use a new scaler with only the common columns to avoid sklearn feature name validation issues
                    # when column sets differ (e.g., after different encoding strategies)
                    temp_scaler = StandardScaler()
                    # Set the scale parameters from the original scaler for matching columns
                    col_indices = [self.scalers['columns'].index(col) for col in scale_cols]
                    temp_scaler.mean_ = self.scalers['standard'].mean_[col_indices]
                    temp_scaler.scale_ = self.scalers['standard'].scale_[col_indices]
                    temp_scaler.var_ = self.scalers['standard'].var_[col_indices]
                    temp_scaler.n_features_in_ = len(scale_cols)
                    df_scaled[scale_cols] = temp_scaler.transform(df_scaled[scale_cols])
        
        return df_scaled
    
    def preprocess_pipeline(self, df, target_col='Default', fit=True):
        """Complete preprocessing pipeline"""
        print("\n" + "="*50)
        print("Starting Preprocessing Pipeline")
        print("="*50)
        
        if fit:
            self.analyze_missing_values(df)
        
        df_processed = self.handle_missing_values(df)
        
        df_processed = self.handle_outliers(df_processed)
        
        if target_col in df_processed.columns:
            df_processed = self.encode_categorical_features(df_processed, target_col, fit=fit)
        else:
            df_processed = self.encode_categorical_features(df_processed, fit=fit)
        
        scale_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if 'ID' in scale_cols:
            scale_cols.remove('ID')
        if target_col in scale_cols:
            scale_cols.remove(target_col)
        
        df_processed = self.scale_features(df_processed, columns=scale_cols, fit=fit)
        
        print("\n" + "="*50)
        print("Preprocessing Complete")
        print(f"Final shape: {df_processed.shape}")
        print("="*50)
        
        return df_processed
