"""
Model Training Script with Visualizations
Trains multiple ML models and generates comparison graphs
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Get project root (parent of train folder)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
TRAIN_DIR = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / 'backend'

def load_dataset() -> pd.DataFrame:
    """Load dataset from data folder"""
    csv_path = DATA_DIR / 'studataset.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    # Create synthetic 'depression' target
    df['depression'] = ((df['failures'] > 0) | (df['absences'] > 15) | (df['G3'] < 8)).astype(int)
    return df

def get_features_and_target(df: pd.DataFrame):
    """Extract features and target"""
    features = ['failures', 'absences', 'goout', 'Dalc', 'Walc', 'studytime', 'health', 'G3']
    target = 'depression'
    X = df[features]
    y = df[target]
    return X, y

def make_numeric_pipeline() -> Pipeline:
    """Create preprocessing pipeline"""
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

def build_model_pipelines():
    """Build model pipelines"""
    preprocessor = ColumnTransformer(
        transformers=[('num', make_numeric_pipeline(), list(range(8)))]
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    pipelines = {name: Pipeline(steps=[('prep', preprocessor), ('clf', model)]) 
                 for name, model in models.items()}
    return pipelines

def plot_accuracy_comparison(scores: dict, save_path: Path):
    """Plot accuracy comparison bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(scores.keys())
    accuracies = [scores[m] * 100 for m in models]  # Convert to percentage
    
    colors = ['#0b5cff', '#22c55e', '#f59e0b', '#ef4444']
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved accuracy comparison chart to {save_path / 'accuracy_comparison.png'}")
    plt.close()

def plot_confusion_matrices(pipelines: dict, X_test, y_test, save_path: Path):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, (name, pipe) in enumerate(pipelines.items()):
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar_kws={'label': 'Count'}, square=True, linewidths=1)
        axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%', 
                           fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
        axes[idx].set_xticklabels(['Low Risk', 'High Risk'])
        axes[idx].set_yticklabels(['Low Risk', 'High Risk'])
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrices to {save_path / 'confusion_matrices.png'}")
    plt.close()

def plot_prediction_distribution(pipelines: dict, X_test, y_test, save_path: Path):
    """Plot prediction probability distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, (name, pipe) in enumerate(pipelines.items()):
        # Get prediction probabilities if available
        if hasattr(pipe.named_steps['clf'], 'predict_proba'):
            y_proba = pipe.predict_proba(X_test)[:, 1]  # Probability of class 1 (high risk)
        else:
            y_proba = pipe.predict(X_test).astype(float)
        
        # Separate by actual class
        low_risk_proba = y_proba[y_test == 0]
        high_risk_proba = y_proba[y_test == 1]
        
        axes[idx].hist(low_risk_proba, bins=20, alpha=0.6, label='Low Risk (Actual)', 
                      color='green', edgecolor='black')
        axes[idx].hist(high_risk_proba, bins=20, alpha=0.6, label='High Risk (Actual)', 
                      color='red', edgecolor='black')
        axes[idx].set_xlabel('Predicted Probability (High Risk)', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].set_title(f'{name}', fontsize=11, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(alpha=0.3)
    
    plt.suptitle('Prediction Probability Distributions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path / 'prediction_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved prediction distributions to {save_path / 'prediction_distributions.png'}")
    plt.close()

def plot_feature_importance(lr_model, feature_names, save_path: Path):
    """Plot feature importance for Logistic Regression"""
    if hasattr(lr_model.named_steps['clf'], 'coef_'):
        coef = lr_model.named_steps['clf'].coef_[0]
        feature_importance = np.abs(coef)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
        bars = ax.barh(feature_names, feature_importance, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, val in zip(bars, feature_importance):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.3f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Feature Importance (|Coefficient|)', fontsize=12, fontweight='bold')
        ax.set_title('Logistic Regression - Feature Importance', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance chart to {save_path / 'feature_importance.png'}")
        plt.close()

def main():
    print("=" * 60)
    print("Model Training and Evaluation Script")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading dataset...")
    df = load_dataset()
    print(f"✓ Loaded {len(df)} samples")
    
    # Prepare features
    print("\n[2/5] Preparing features...")
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"✓ Features: {list(X.columns)}")
    
    # Build models
    print("\n[3/5] Building and training models...")
    pipelines = build_model_pipelines()
    scores = {}
    
    for name, pipe in pipelines.items():
        print(f"  Training {name}...", end=' ')
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores[name] = round(float(acc), 4)
        print(f"✓ Accuracy: {acc*100:.2f}%")
    
    # Save metrics
    print("\n[4/5] Saving results...")
    metrics_dict = {
        'logistic_regression': scores['Logistic Regression'],
        'naive_bayes': scores['Naive Bayes'],
        'svm_rbf': scores['SVM (RBF)'],
        'knn': scores['KNN']
    }
    
    # Save metrics to backend folder (for API)
    metrics_path = BACKEND_DIR / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"✓ Saved metrics to {metrics_path}")
    
    # Save Logistic Regression model to backend (for API)
    lr_model = pipelines['Logistic Regression']
    model_path = BACKEND_DIR / 'model.pkl'
    joblib.dump(lr_model, model_path)
    print(f"✓ Saved Logistic Regression model to {model_path}")
    
    # Generate visualizations
    print("\n[5/5] Generating visualizations...")
    plot_accuracy_comparison(scores, TRAIN_DIR)
    plot_confusion_matrices(pipelines, X_test, y_test, TRAIN_DIR)
    plot_prediction_distribution(pipelines, X_test, y_test, TRAIN_DIR)
    plot_feature_importance(lr_model, list(X.columns), TRAIN_DIR)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nModel Accuracies:")
    for name, acc in scores.items():
        print(f"  {name:20s}: {acc*100:6.2f}%")
    print(f"\nVisualizations saved to: {TRAIN_DIR}")
    print(f"Model saved to: {model_path}")
    print("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

