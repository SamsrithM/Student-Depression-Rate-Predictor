"""Train and save a Logistic Regression model (one-file runner).

Usage:
  python train/logistic_regression.py

This script trains a logistic regression pipeline, saves the model to
`backend/model_logistic_regression.pkl` and updates `backend/metrics.json`.
"""
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Support running as a module (python -m train.logistic_regression)
# and running directly (python train/logistic_regression.py).
try:
    from train.utils import load_dataset, get_features_and_target, build_preprocessor, update_metrics, BACKEND_DIR
except ModuleNotFoundError:
    # when the script is executed directly, the package root may not be on sys.path
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from train.utils import load_dataset, get_features_and_target, build_preprocessor, update_metrics, BACKEND_DIR


MODEL_KEY = 'logistic_regression'


def main():
    PROJECT_TRAIN_DIR = Path(__file__).parent
    df = load_dataset()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    pipeline = Pipeline(steps=[('prep', build_preprocessor()), ('clf', clf)])
    pipeline.fit(X_train, y_train)

    # evaluate
    acc = pipeline.score(X_test, y_test)
    print(f"Logistic Regression accuracy: {acc*100:.2f}%")

    # Plot confusion matrix and prediction distribution
    sns.set_style('darkgrid')
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Logistic Regression Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # predicted probability distribution (if available)
    if hasattr(pipeline.named_steps['clf'], 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipeline.predict(X_test).astype(float)

    axes[1].hist([y_proba[y_test == 0], y_proba[y_test == 1]], bins=20,
                 label=['Low Risk (Actual)', 'High Risk (Actual)'], color=['green', 'red'], alpha=0.6)
    axes[1].set_title('Predicted Probability Distribution')
    axes[1].set_xlabel('Predicted Probability (class 1)')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    # save model
    BACKEND_DIR.mkdir(parents=True, exist_ok=True)
    model_path = BACKEND_DIR / 'model_logistic_regression.pkl'
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")

    # update metrics
    update_metrics(MODEL_KEY, float(acc))
    print(f"Updated metrics key '{MODEL_KEY}' in backend/metrics.json")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
