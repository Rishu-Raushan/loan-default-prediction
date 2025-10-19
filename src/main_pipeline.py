import sys
import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator


def load_config(config_path='config/config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main pipeline execution"""
    print("="*70)
    print("LOAN DEFAULT PREDICTION - COMPLETE PIPELINE")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports/plots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    config = load_config()
    print("\nConfiguration loaded successfully")
    
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING")
    print("="*70)
    
    preprocessor = DataPreprocessor(config)
    df = preprocessor.load_data(config['data']['raw_data_path'])
    
    target_col = 'Default' if 'Default' in df.columns else 'loan_status'
    print(f"\nTarget column: {target_col}")
    
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)
    
    df_processed = preprocessor.preprocess_pipeline(df, target_col=target_col, fit=True)
    
    print("\n" + "="*70)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*70)
    
    feature_engineer = FeatureEngineer(config)
    df_featured = feature_engineer.feature_engineering_pipeline(
        df_processed, target_col=target_col, fit=True
    )
    
    print("\n" + "="*70)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("="*70)
    
    if 'ID' in df_featured.columns:
        X = df_featured.drop(columns=[target_col, 'ID'])
    else:
        X = df_featured.drop(columns=[target_col])
    y = df_featured[target_col]
    
    if feature_engineer.selected_features:
        available_features = [f for f in feature_engineer.selected_features if f in X.columns]
        X = X[available_features]
        print(f"\nUsing {len(available_features)} selected features")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['data']['test_size'], 
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"\nTraining set class distribution:")
    print(pd.Series(y_train).value_counts())
    print(f"\nTest set class distribution:")
    print(pd.Series(y_test).value_counts())
    
    print("\n" + "="*70)
    print("STEP 5: MODEL TRAINING")
    print("="*70)
    
    trainer = ModelTrainer(config)
    training_results = trainer.train_all_models(X_train, y_train)
    
    if config['hyperparameter_tuning']['enable']:
        print("\n" + "="*70)
        print("STEP 6: HYPERPARAMETER TUNING")
        print("="*70)
        
        best_model_name = trainer.best_model_name
        tuned_model, best_params = trainer.hyperparameter_tuning(
            X_train, y_train, best_model_name
        )
        trainer.best_model = tuned_model
        trainer.best_model_name = f"{best_model_name}_tuned"
    
    print("\n" + "="*70)
    print("STEP 7: MODEL EVALUATION")
    print("="*70)
    
    evaluator = ModelEvaluator(config)
    
    for model_name, model_info in trainer.models.items():
        model = model_info['model']
        metrics = evaluator.evaluate_model(model, X_test, y_test, model_name)
        
        evaluator.plot_roc_curve(
            model, X_test, y_test, model_name,
            save_path=f'reports/plots/roc_curve_{model_name}.png'
        )
        
        evaluator.plot_precision_recall_curve(
            model, X_test, y_test, model_name,
            save_path=f'reports/plots/pr_curve_{model_name}.png'
        )
        
        if model_name in evaluator.results:
            cm = evaluator.results[model_name]['confusion_matrix']
            evaluator.plot_confusion_matrix(
                cm, model_name,
                save_path=f'reports/plots/confusion_matrix_{model_name}.png'
            )
        
        evaluator.plot_feature_importance(
            model, X.columns, top_n=20,
            save_path=f'reports/plots/feature_importance_{model_name}.png'
        )
    
    comparison_df = evaluator.compare_models(save_path='reports/model_comparison.csv')
    
    print("\n" + "="*70)
    print("STEP 8: THRESHOLD OPTIMIZATION")
    print("="*70)
    
    best_model = trainer.best_model
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    if config['evaluation']['threshold_optimization']:
        cost_fn_ratio = config['evaluation']['business_cost_fn_ratio']
        optimal_threshold, threshold_results = evaluator.optimize_threshold(
            y_test, y_pred_proba, cost_fn_ratio=cost_fn_ratio
        )
        
        threshold_results.to_csv('reports/threshold_optimization.csv', index=False)
    else:
        optimal_threshold = 0.5
    
    print("\n" + "="*70)
    print("STEP 9: BUSINESS INSIGHTS")
    print("="*70)
    
    business_insights = evaluator.generate_business_insights(
        best_model, X_test, y_test, threshold=optimal_threshold
    )
    
    insights_df = pd.DataFrame([business_insights])
    insights_df.to_csv('reports/business_insights.csv', index=False)
    
    print("\n" + "="*70)
    print("STEP 10: MODEL PERSISTENCE")
    print("="*70)
    
    import joblib
    
    trainer.save_model(best_model, f'models/best_model_{trainer.best_model_name}.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(feature_engineer, 'models/feature_engineer.pkl')
    
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    metadata = {
        'model_name': trainer.best_model_name,
        'features': feature_names,
        'optimal_threshold': optimal_threshold,
        'test_performance': {
            'roc_auc': evaluator.results[trainer.best_model_name]['roc_auc'],
            'precision': evaluator.results[trainer.best_model_name]['precision'],
            'recall': evaluator.results[trainer.best_model_name]['recall'],
            'f1': evaluator.results[trainer.best_model_name]['f1']
        }
    }
    
    import json
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("\nModel artifacts saved:")
    print("- models/best_model_{}.pkl".format(trainer.best_model_name))
    print("- models/preprocessor.pkl")
    print("- models/feature_engineer.pkl")
    print("- models/feature_names.pkl")
    print("- models/model_metadata.json")
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"ROC-AUC Score: {evaluator.results[trainer.best_model_name]['roc_auc']:.4f}")
    print(f"Precision: {evaluator.results[trainer.best_model_name]['precision']:.4f}")
    print(f"Recall: {evaluator.results[trainer.best_model_name]['recall']:.4f}")
    print(f"F1-Score: {evaluator.results[trainer.best_model_name]['f1']:.4f}")
    print(f"\nOptimal Threshold: {optimal_threshold:.2f}")
    print(f"\nDefault Detection Rate: {business_insights['default_detection_rate']:.2f}%")
    print(f"False Positive Rate: {business_insights['false_positive_rate']:.2f}%")
    
    print("\nAll reports and visualizations saved to reports/")
    print("\nTo deploy the model, run: python deployment/app.py")


if __name__ == "__main__":
    main()
