import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import joblib

# 1. Define columns
categorical_cols = ["Land Cover", "Protected Area Designation", "Nearest Grid Type"]
binary_col = "In SPUG Area?"

# 2. yes/no to binary converter
def yes_no_to_binary(X):
    # Works for DataFrame or numpy array
    X = pd.DataFrame(X, columns=[binary_col]) if not isinstance(X, pd.DataFrame) else X
    return (X == "Yes").astype(int)

binary_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("yesno", FunctionTransformer(yes_no_to_binary, validate=False))
])

# 3. Numeric columns (exclude categorical and binary)
numeric_cols = [col for col in X_supervised_df.columns 
                if col not in categorical_cols + [binary_col]]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 4. Column transformer
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
    ("bin", binary_transformer, [binary_col])
])

# 5. Full pipeline
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", SVC(
        kernel=best_params['svm_kernel'], 
        C=best_params["svm_C"],  
        class_weight="balanced",
        probability=True
    ))
])

# 6. Fit and save
pipeline.fit(X_supervised_df, y_supervised_df)