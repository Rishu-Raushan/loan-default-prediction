import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Model training class supporting multiple algorithms with
    hyperparameter tuning and class imbalance handling
    """
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def handle_imbalance(self, X, y, strategy='smote'):
        """Handle class imbalance using SMOTE or other techniques"""
        print(f"\nHandling class imbalance using {strategy}...")
        print(f"Original class distribution:\n{pd.Series(y).value_counts()}")
        
        if strategy == 'smote':
            sampling_strategy = self.config['model'].get('smote_sampling_strategy', 0.5)
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif strategy == 'smote_tomek':
            smote_tomek = SMOTETomek(random_state=42)
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
        
        print(f"Resampled class distribution:\n{pd.Series(y_resampled).value_counts()}")
        
        return X_resampled, y_resampled
    
    def get_model(self, model_name, params=None):
        """Initialize model based on name"""
        if model_name == 'logistic_regression':
            model = LogisticRegression(random_state=42, **(params or {}))
        elif model_name == 'random_forest':
            model = RandomForestClassifier(random_state=42, n_jobs=-1, **(params or {}))
        elif model_name == 'xgboost':
            model = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, 
                                eval_metric='logloss', **(params or {}))
        elif model_name == 'lightgbm':
            model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **(params or {}))
        elif model_name == 'catboost':
            model = CatBoostClassifier(random_state=42, verbose=0, **(params or {}))
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    def train_model(self, X_train, y_train, model_name, params=None):
        """Train a single model"""
        print(f"\nTraining {model_name}...")
        
        model = self.get_model(model_name, params)
        model.fit(X_train, y_train)
        
        cv = StratifiedKFold(n_splits=self.config['model']['n_folds'], shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        print(f"{model_name} - CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.models[model_name] = {
            'model': model,
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean()
        }
        
        return model, cv_scores.mean()
    
    def train_all_models(self, X_train, y_train):
        """Train all configured models"""
        print("\n" + "="*50)
        print("Training All Models")
        print("="*50)
        
        if self.config['model']['handle_imbalance']:
            strategy = self.config['model']['imbalance_strategy']
            X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train, strategy)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        results = []
        
        for model_config in self.config['models_to_train']:
            model_name = model_config['name']
            params = model_config.get('params', {})
            
            model, cv_score = self.train_model(X_train_balanced, y_train_balanced, 
                                              model_name, params)
            
            results.append({
                'model_name': model_name,
                'cv_roc_auc': cv_score
            })
        
        results_df = pd.DataFrame(results).sort_values('cv_roc_auc', ascending=False)
        print("\n" + "="*50)
        print("Model Training Results")
        print("="*50)
        print(results_df.to_string(index=False))
        
        best_model_name = results_df.iloc[0]['model_name']
        self.best_model = self.models[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name}")
        
        return results_df
    
    def hyperparameter_tuning(self, X_train, y_train, model_name):
        """Hyperparameter tuning using Optuna"""
        print(f"\nTuning hyperparameters for {model_name}...")
        
        if self.config['model']['handle_imbalance']:
            strategy = self.config['model']['imbalance_strategy']
            X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train, strategy)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 5)
                }
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
                }
            else:
                return 0
            
            model = self.get_model(model_name, params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                    cv=cv, scoring='roc_auc', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, 
                      n_trials=self.config['hyperparameter_tuning']['n_trials'],
                      timeout=self.config['hyperparameter_tuning']['timeout'],
                      show_progress_bar=True)
        
        print(f"\nBest parameters: {study.best_params}")
        print(f"Best CV ROC-AUC: {study.best_value:.4f}")
        
        best_model = self.get_model(model_name, study.best_params)
        best_model.fit(X_train_balanced, y_train_balanced)
        
        self.models[f'{model_name}_tuned'] = {
            'model': best_model,
            'params': study.best_params,
            'cv_score': study.best_value
        }
        
        return best_model, study.best_params
    
    def save_model(self, model, filepath):
        """Save trained model to disk"""
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from disk"""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
