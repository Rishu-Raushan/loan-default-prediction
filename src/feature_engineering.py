import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering class for creating domain-specific features
    and performing feature selection
    """
    
    def __init__(self, config):
        self.config = config
        self.selected_features = None
        self.feature_importance = None
        
    def create_domain_features(self, df):
        """Create domain-specific features based on loan data"""
        print("\nCreating domain-specific features...")
        df_fe = df.copy()
        
        if 'Client_Income' in df_fe.columns and 'Credit_Amount' in df_fe.columns:
            df_fe['Income_to_Credit_Ratio'] = df_fe['Client_Income'] / (df_fe['Credit_Amount'] + 1)
            df_fe['Income_to_Credit_Ratio'] = df_fe['Income_to_Credit_Ratio'].fillna(0)
            print("Created: Income_to_Credit_Ratio")
        
        if 'Loan_Annuity' in df_fe.columns and 'Client_Income' in df_fe.columns:
            df_fe['Annuity_Income_Ratio'] = df_fe['Loan_Annuity'] / (df_fe['Client_Income'] + 1)
            df_fe['Annuity_Income_Ratio'] = df_fe['Annuity_Income_Ratio'].fillna(0)
            print("Created: Annuity_Income_Ratio")
        
        if 'Credit_Amount' in df_fe.columns and 'Loan_Annuity' in df_fe.columns:
            df_fe['Credit_to_Annuity_Ratio'] = df_fe['Credit_Amount'] / (df_fe['Loan_Annuity'] + 1)
            df_fe['Credit_to_Annuity_Ratio'] = df_fe['Credit_to_Annuity_Ratio'].fillna(0)
            print("Created: Credit_to_Annuity_Ratio")
        
        if 'Age_Days' in df_fe.columns:
            df_fe['Age_Years'] = abs(df_fe['Age_Days']) / 365.25
            df_fe['Age_Years'] = df_fe['Age_Years'].fillna(df_fe['Age_Years'].median())
            df_fe['Age_Group'] = pd.cut(df_fe['Age_Years'], 
                                        bins=[0, 25, 35, 45, 55, 100], 
                                        labels=[0, 1, 2, 3, 4])
            df_fe['Age_Group'] = df_fe['Age_Group'].astype(float).fillna(2)
            print("Created: Age_Years, Age_Group")
        
        if 'Employed_Days' in df_fe.columns:
            df_fe['Employment_Years'] = abs(df_fe['Employed_Days']) / 365.25
            df_fe['Employment_Years'] = df_fe['Employment_Years'].fillna(0)
            df_fe['Employment_Stability'] = np.where(df_fe['Employment_Years'] > 5, 1, 0)
            print("Created: Employment_Years, Employment_Stability")
        
        if 'Child_Count' in df_fe.columns and 'Client_Family_Members' in df_fe.columns:
            df_fe['Has_Children'] = np.where(df_fe['Child_Count'] > 0, 1, 0)
            df_fe['Adult_Family_Members'] = df_fe['Client_Family_Members'] - df_fe['Child_Count']
            df_fe['Adult_Family_Members'] = df_fe['Adult_Family_Members'].fillna(0)
            df_fe['Family_Size_Category'] = pd.cut(df_fe['Client_Family_Members'], 
                                                     bins=[0, 2, 4, 10], 
                                                     labels=[0, 1, 2])
            df_fe['Family_Size_Category'] = df_fe['Family_Size_Category'].astype(float).fillna(0)
            print("Created: Has_Children, Adult_Family_Members, Family_Size_Category")
        
        if 'Car_Owned' in df_fe.columns and 'House_Own' in df_fe.columns:
            df_fe['Total_Assets'] = df_fe['Car_Owned'] + df_fe['House_Own']
            df_fe['Asset_Owner'] = np.where(df_fe['Total_Assets'] > 0, 1, 0)
            print("Created: Total_Assets, Asset_Owner")
        
        if 'Mobile_Tag' in df_fe.columns and 'Homephone_Tag' in df_fe.columns and 'Workphone_Working' in df_fe.columns:
            df_fe['Total_Phone_Contacts'] = df_fe['Mobile_Tag'] + df_fe['Homephone_Tag'] + df_fe['Workphone_Working']
            print("Created: Total_Phone_Contacts")
        
        if 'Score_Source_2' in df_fe.columns and 'Score_Source_3' in df_fe.columns:
            df_fe['Average_Score'] = (df_fe['Score_Source_2'] + df_fe['Score_Source_3']) / 2
            df_fe['Average_Score'] = df_fe['Average_Score'].fillna(df_fe['Average_Score'].median())
            df_fe['Max_Score'] = df_fe[['Score_Source_2', 'Score_Source_3']].max(axis=1)
            df_fe['Min_Score'] = df_fe[['Score_Source_2', 'Score_Source_3']].min(axis=1)
            df_fe['Score_Range'] = df_fe['Max_Score'] - df_fe['Min_Score']
            df_fe['Score_Range'] = df_fe['Score_Range'].fillna(0)
            print("Created: Average_Score, Max_Score, Min_Score, Score_Range")
        
        if 'Registration_Days' in df_fe.columns:
            df_fe['Registration_Years'] = abs(df_fe['Registration_Days']) / 365.25
            df_fe['Registration_Years'] = df_fe['Registration_Years'].fillna(0)
            df_fe['Recent_Registration'] = np.where(df_fe['Registration_Years'] < 1, 1, 0)
            print("Created: Registration_Years, Recent_Registration")
        
        if 'ID_Days' in df_fe.columns:
            df_fe['ID_Years'] = abs(df_fe['ID_Days']) / 365.25
            df_fe['ID_Years'] = df_fe['ID_Years'].fillna(0)
            df_fe['Recent_ID_Change'] = np.where(df_fe['ID_Years'] < 1, 1, 0)
            print("Created: ID_Years, Recent_ID_Change")
        
        if 'Phone_Change' in df_fe.columns:
            df_fe['Phone_Change_Years'] = abs(df_fe['Phone_Change']) / 365.25
            df_fe['Phone_Change_Years'] = df_fe['Phone_Change_Years'].fillna(0)
            df_fe['Recent_Phone_Change'] = np.where(df_fe['Phone_Change_Years'] < 1, 1, 0)
            print("Created: Phone_Change_Years, Recent_Phone_Change")
        
        if 'Application_Process_Day' in df_fe.columns:
            df_fe['Weekend_Application'] = np.where(df_fe['Application_Process_Day'].isin([0, 6]), 1, 0)
            print("Created: Weekend_Application")
        
        if 'Application_Process_Hour' in df_fe.columns:
            df_fe['Business_Hours_Application'] = np.where(
                (df_fe['Application_Process_Hour'] >= 9) & (df_fe['Application_Process_Hour'] <= 17), 1, 0
            )
            print("Created: Business_Hours_Application")
        
        if 'Client_Permanent_Match_Tag' in df_fe.columns and 'Client_Contact_Work_Tag' in df_fe.columns:
            df_fe['Address_Mismatch_Count'] = df_fe['Client_Permanent_Match_Tag'] + df_fe['Client_Contact_Work_Tag']
            print("Created: Address_Mismatch_Count")
        

        
        if 'Credit_Bureau' in df_fe.columns:
            df_fe['High_Credit_Inquiries'] = np.where(df_fe['Credit_Bureau'] > 5, 1, 0)
            print("Created: High_Credit_Inquiries")
        
        if 'Client_Income' in df_fe.columns and 'Client_Family_Members' in df_fe.columns:
            df_fe['Income_Per_Family_Member'] = df_fe['Client_Income'] / (df_fe['Client_Family_Members'] + 1)
            df_fe['Income_Per_Family_Member'] = df_fe['Income_Per_Family_Member'].fillna(0)
            print("Created: Income_Per_Family_Member")
        
        print(f"\nTotal features after engineering: {df_fe.shape[1]}")
        
        print("\nChecking for remaining NaN values...")
        nan_cols = df_fe.columns[df_fe.isnull().any()].tolist()
        if nan_cols:
            print(f"Warning: NaN values found in {len(nan_cols)} columns, filling with 0")
            df_fe = df_fe.fillna(0)
        
        return df_fe
    
    def create_interaction_features(self, df):
        """Create interaction features between important variables"""
        print("\nCreating interaction features...")
        df_interact = df.copy()
        
        interactions = [
            ('Client_Income', 'Credit_Amount'),
            ('Age_Years', 'Employment_Years'),
            ('Total_Assets', 'Client_Income'),
        ]
        
        for feat1, feat2 in interactions:
            if feat1 in df_interact.columns and feat2 in df_interact.columns:
                df_interact[f'{feat1}_x_{feat2}'] = df_interact[feat1] * df_interact[feat2]
                df_interact[f'{feat1}_x_{feat2}'] = df_interact[f'{feat1}_x_{feat2}'].fillna(0)
                print(f"Created: {feat1}_x_{feat2}")
        
        return df_interact
    
    def select_features_importance(self, X, y, n_features=30):
        """Select features using Random Forest feature importance"""
        print(f"\nSelecting top {n_features} features using importance...")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance
        top_features = feature_importance.head(n_features)['feature'].tolist()
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return top_features
    
    def select_features_statistical(self, X, y, n_features=30):
        """Select features using statistical tests"""
        print(f"\nSelecting top {n_features} features using statistical tests...")
        
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X, y)
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        top_features = feature_scores.head(n_features)['feature'].tolist()
        
        print("\nTop 10 Features by Statistical Score:")
        print(feature_scores.head(10).to_string(index=False))
        
        return top_features
    
    def feature_engineering_pipeline(self, df, target_col='Default', fit=True):
        """Complete feature engineering pipeline"""
        print("\n" + "="*50)
        print("Starting Feature Engineering Pipeline")
        print("="*50)
        
        df_fe = self.create_domain_features(df)
        
        if self.config['feature_engineering']['create_interaction_features']:
            df_fe = self.create_interaction_features(df_fe)
        
        if fit and target_col in df_fe.columns:
            if 'ID' in df_fe.columns:
                X = df_fe.drop(columns=[target_col, 'ID'])
            else:
                X = df_fe.drop(columns=[target_col])
            y = df_fe[target_col]
            
            n_features = self.config['feature_engineering'].get('n_features_to_select', 30)
            n_features = min(n_features, X.shape[1])
            
            self.selected_features = self.select_features_importance(X, y, n_features)
        
        print("\n" + "="*50)
        print("Feature Engineering Complete")
        print(f"Final shape: {df_fe.shape}")
        print("="*50)
        
        return df_fe
