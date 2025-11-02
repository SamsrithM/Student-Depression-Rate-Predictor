# Model Training Scripts

This folder contains scripts for training and evaluating machine learning models.

## Files

- `train_models.py` - Main training script with visualizations

## Usage

### Train all models and generate visualizations:

```bash
python train/train_models.py
```

This will:
1. Load data from `data/studataset.csv`
2. Train 4 models: Logistic Regression, Naive Bayes, SVM, KNN
3. Generate comparison graphs using matplotlib
4. Save the best model (Logistic Regression) to `backend/model.pkl`
5. Save metrics to `backend/metrics.json`

## Generated Visualizations

After running the script, you'll find these graphs in the `train/` folder:

1. **accuracy_comparison.png** - Bar chart comparing all model accuracies
2. **confusion_matrices.png** - Confusion matrices for all 4 models
3. **prediction_distributions.png** - Probability distributions for predictions
4. **feature_importance.png** - Feature importance from Logistic Regression

## Requirements

Make sure you have installed:
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

Install with: `pip install -r requirements.txt`
 
---

## Per-model scripts (new)

I split the original `train_models.py` into individual, runnable scripts so you
can train each model separately. New files added:

- `utils.py` - shared utilities (data loading, preprocessing, metrics helper)
- `logistic_regression.py` - trains Logistic Regression
- `naive_bayes.py` - trains Gaussian Naive Bayes
- `svm_rbf.py` - trains SVM (RBF)
- `knn.py` - trains K-Nearest Neighbors

How to run (from repository root)

PowerShell / Windows:
```powershell
python train\logistic_regression.py
python train\naive_bayes.py
python train\svm_rbf.py
python train\knn.py
```

Or with module invocation (cross-platform):
```powershell
python -m train.logistic_regression
python -m train.naive_bayes
python -m train.svm_rbf
python -m train.knn
```

When you run a script it will:
- load `data/studataset.csv` and create the `depression` target
- preprocess features (median impute + standard scaling)
- train the selected model
- save the model to `backend/model_<model_name>.pkl`
- update `backend/metrics.json` with the accuracy value under a
	model-specific key (e.g. `logistic_regression`).

Notes
- Ensure dependencies in `requirements.txt` are installed.
- If `backend/` or `backend/metrics.json` do not exist they will be created.
- The scripts are intentionally small so you can run and debug each model
	independently.

