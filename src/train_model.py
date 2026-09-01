import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

FEATURES = ['follower_following_ratio', 'completeness_score', 'posts_per_follower',
            'username_digit_ratio', 'fullname_digit_ratio', 'fullname_word_count',
            'name_equals_username']

def train():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('data/processed_accounts.csv')
    X = df[FEATURES]
    y = df['fake']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

    # Baseline model
    lr = LogisticRegression()
    lr.fit(X_res, y_res)
    lr_preds = lr.predict(X_test_scaled)
    print("Logistic Regression Report:\n", classification_report(y_test, lr_preds))

      # Main model - hyperparameter tuning via grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_search.fit(X_res, y_res)

    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation ROC-AUC:", grid_search.best_score_)

    rf = grid_search.best_estimator_
    preds = rf.predict(X_test_scaled)
    probs = rf.predict_proba(X_test_scaled)[:, 1]

    
  

    print("Random Forest Report:\n", classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    # Confusion matrix
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Random Forest')
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label='Random Forest')
    plt.plot([0, 1], [0, 1], '--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('reports/roc_curve.png')
    plt.close()

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    sns.barplot(x=importances.values, y=importances.index)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    plt.close()

    joblib.dump(rf, 'models/rf_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved to models/")
    print("Evaluation charts saved to reports/")

if __name__ == "__main__":
    train()
