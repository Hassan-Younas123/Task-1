import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

df = pd.read_csv(r"C:\Users\Hassan\PyCharmMiscProject\customer_churn_dataset-testing-master.csv")

df = df.drop(columns=["CustomerID"])

df = pd.get_dummies(df, columns=["Gender", "Subscription Type", "Contract Length"], drop_first=True)

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train_scaled, y_train)
baseline_val_pred = baseline.predict(X_val_scaled)

rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_val_pred = rf.predict(X_val)

def evaluate(name, y_true, y_pred, y_proba=None):
    result = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    if y_proba is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_proba)
    return result

results = []
results.append(evaluate("Logistic Regression (baseline)", y_val, baseline_val_pred, baseline.predict_proba(X_val_scaled)[:, 1]))
results.append(evaluate("Random Forest", y_val, rf_val_pred, rf.predict_proba(X_val)[:, 1]))

results_df = pd.DataFrame(results)
print(results_df)

rf_test_pred = rf.predict(X_test)
rf_test_proba = rf.predict_proba(X_test)[:, 1]

print("\nTest set performance (Random Forest):")
print(evaluate("Random Forest (test)", y_test, rf_test_pred, rf_test_proba))
print(classification_report(y_test, rf_test_pred))

cm = confusion_matrix(y_test, rf_test_pred)
print("Confusion matrix:\n", cm)

errors = X_test.copy()
errors["actual"] = y_test.values
errors["predicted"] = rf_test_pred
errors["proba"] = rf_test_proba
false_negatives = errors[(errors.actual == 1) & (errors.predicted == 0)]
false_positives = errors[(errors.actual == 0) & (errors.predicted == 1)]
print(f"\nFalse negatives: {len(false_negatives)}, False positives: {len(false_positives)}")

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:\n", importances)

model_card = {
    "model_type": "RandomForestClassifier",
    "n_estimators": 300,
    "max_depth": 10,
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
    "test_roc_auc": roc_auc_score(y_test, rf_test_proba),
    "top_features": importances.head(5).to_dict(),
    "known_limitations": "Dataset does not include internet service type or contract auto-renewal flags, and 'Last Interaction' is a coarse recency proxy rather than a true engagement signal."
}
print("\nModel card:\n", model_card)
