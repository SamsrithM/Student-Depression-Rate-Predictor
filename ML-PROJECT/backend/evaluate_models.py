import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def load_dataset() -> pd.DataFrame:
    # Use the same dataset path as trainer
    csv_path = Path('d:/study college/3rd year/ML-PROJECT/studataset.csv')
    df = pd.read_csv(csv_path)
    # Create the synthetic target like in train_model.py
    df['depression'] = ((df['failures'] > 0) | (df['absences'] > 15) | (df['G3'] < 8)).astype(int)
    return df


def get_features_and_target(df: pd.DataFrame):
    features = ['failures', 'absences', 'goout', 'Dalc', 'Walc', 'studytime', 'health', 'G3']
    target = 'depression'
    X = df[features]
    y = df[target]
    return X, y


def make_numeric_pipeline() -> Pipeline:
    # Numeric pipeline: impute then scale where appropriate
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def build_model_pipelines():
    preprocessor = ColumnTransformer(
        transformers=[('num', make_numeric_pipeline(), list(range(8)))],
        remainder='drop'
    )

    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'naive_bayes': GaussianNB(),
        'svm_rbf': SVC(kernel='rbf', probability=False, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5)
    }

    pipelines = {name: Pipeline(steps=[('prep', preprocessor), ('clf', model)]) for name, model in models.items()}
    return pipelines


def main():
    df = load_dataset()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipelines = build_model_pipelines()
    scores = {}

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        scores[name] = round(float(acc), 4)

    # Save a copy and also print
    out_path = Path(__file__).parent / 'metrics.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2)

    # Also save the LR model like existing flow so API can use it if desired
    if 'logistic_regression' in pipelines:
        joblib.dump(pipelines['logistic_regression'], (Path(__file__).parent / 'model.pkl').as_posix())

    print('Test accuracies:')
    for k, v in scores.items():
        print(f'- {k}: {v}')
    print(f"Saved metrics to {out_path}")


if __name__ == '__main__':
    main()



