import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, 
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Model evaluation class for comprehensive performance analysis
    """
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """Comprehensive model evaluation"""
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        print(f"\nPerformance Metrics:")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"PR-AUC Score: {pr_auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        print(f"\nSpecificity: {specificity:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        
        self.results[model_name] = {
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return {
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pr_auc': pr_auc
        }
    
    def plot_roc_curve(self, model, X_test, y_test, model_name='Model', save_path=None):
        """Plot ROC curve"""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        plt.close()
    
    def plot_precision_recall_curve(self, model, X_test, y_test, model_name='Model', save_path=None):
        """Plot Precision-Recall curve"""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, cm, model_name='Model', save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Non-Default', 'Default'],
                   yticklabels=['Non-Default', 'Default'])
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None):
        """Plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("Model does not have feature importance attribute")
            return
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        plt.close()
        
        return feature_importance_df
    
    def calculate_shap_values(self, model, X_sample, model_name='Model', save_path=None):
        """Calculate and plot SHAP values"""
        print(f"\nCalculating SHAP values for {model_name}...")
        
        try:
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"SHAP summary plot saved to {save_path}")
                plt.close()
                
                return shap_values
        except Exception as e:
            print(f"Error calculating SHAP values: {str(e)}")
            return None
    
    def optimize_threshold(self, y_test, y_pred_proba, cost_fn_ratio=5):
        """Optimize classification threshold based on business cost"""
        print(f"\nOptimizing classification threshold (FN cost / FP cost = {cost_fn_ratio})...")
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_cost = float('inf')
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            cost = fp + (cost_fn_ratio * fn)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'cost': cost,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fp': fp,
                'fn': fn
            })
            
            if cost < best_cost:
                best_cost = cost
                best_threshold = threshold
        
        results_df = pd.DataFrame(results)
        
        print(f"\nOptimal threshold: {best_threshold:.2f}")
        print(f"Minimum cost: {best_cost:.0f}")
        
        best_row = results_df[results_df['threshold'] == best_threshold].iloc[0]
        print(f"Precision: {best_row['precision']:.4f}")
        print(f"Recall: {best_row['recall']:.4f}")
        print(f"F1-Score: {best_row['f1']:.4f}")
        
        return best_threshold, results_df
    
    def compare_models(self, save_path=None):
        """Compare all evaluated models"""
        if not self.results:
            print("No models to compare")
            return
        
        comparison_df = pd.DataFrame([
            {
                'Model': name,
                'ROC-AUC': metrics['roc_auc'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'PR-AUC': metrics['pr_auc']
            }
            for name, metrics in self.results.items()
        ]).sort_values('ROC-AUC', ascending=False)
        
        print("\n" + "="*50)
        print("Model Comparison")
        print("="*50)
        print(comparison_df.to_string(index=False))
        
        if save_path:
            comparison_df.to_csv(save_path, index=False)
            print(f"\nComparison saved to {save_path}")
        
        return comparison_df
    
    def generate_business_insights(self, model, X_test, y_test, threshold=0.5):
        """Generate business insights from model predictions"""
        print("\n" + "="*50)
        print("Business Insights")
        print("="*50)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total_loans = len(y_test)
        actual_defaults = y_test.sum()
        predicted_defaults = y_pred.sum()
        
        print(f"\nLoan Portfolio Analysis:")
        print(f"Total loans evaluated: {total_loans:,}")
        print(f"Actual defaults: {actual_defaults:,} ({actual_defaults/total_loans*100:.2f}%)")
        print(f"Predicted defaults: {predicted_defaults:,} ({predicted_defaults/total_loans*100:.2f}%)")
        
        print(f"\nModel Performance:")
        print(f"Correctly identified defaults (True Positives): {tp:,}")
        print(f"Missed defaults (False Negatives): {fn:,}")
        print(f"False alarms (False Positives): {fp:,}")
        print(f"Correctly identified non-defaults (True Negatives): {tn:,}")
        
        print(f"\nRisk Mitigation:")
        default_detection_rate = tp / actual_defaults * 100 if actual_defaults > 0 else 0
        print(f"Default detection rate: {default_detection_rate:.2f}%")
        
        false_positive_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
        print(f"False positive rate: {false_positive_rate:.2f}%")
        
        return {
            'total_loans': total_loans,
            'actual_defaults': actual_defaults,
            'predicted_defaults': predicted_defaults,
            'true_positives': tp,
            'false_negatives': fn,
            'false_positives': fp,
            'true_negatives': tn,
            'default_detection_rate': default_detection_rate,
            'false_positive_rate': false_positive_rate
        }
