from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import pandas as pd
from sklearn.linear_model import LogisticRegression # for L2 regularization
from sklearn.preprocessing import StandardScaler # for L2 regularization

# fetch dataset 
glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759) 
  
# data (as pandas dataframes) 
X = glioma_grading_clinical_and_mutation_features.data.features 
y = glioma_grading_clinical_and_mutation_features.data.targets 
  
# metadata 
print(glioma_grading_clinical_and_mutation_features.metadata) 
  
# variable information 
print(glioma_grading_clinical_and_mutation_features.variables) 


# inspect GBM and LGG class imbalance and visualise in a pie chart
gbm_count = (y["Grade"] == 1).sum()
lgg_count = (y["Grade"] == 0).sum()
print("Number of GBM cases: " + str(gbm_count))
print("Number of LGG cases: " + str(lgg_count))
labels = ["Glioblastoma Multiforme (GBM)", "Lower-Grade Glioma (LGG)"]
sizes = [gbm_count, lgg_count]
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title("Distribution of Glioma Tumor Grades")
plt.show()

# Imbalance Ratio (IR); Rule of thumb IR<1.5: usually not a problem, IR>2: may start causing issues in some models, IR>5: strong imbalance → must be addressed 
majority_class_count = lgg_count
minority_class_count = gbm_count
IR = majority_class_count/minority_class_count
if IR > 5:
    result = "this dataset has a strong class imbalance, therefore must be addressed."
elif IR > 2:
    result = "this dataset has a slight class imbalance and may start causing issues in some models."
else:
    result = "this dataset has a good class balance."
        
print("The imbalance ratio is ", str(round(IR, 2)), ", ", result)

# Chi-Square Test for Balance
obs = [gbm_count, lgg_count]
exp = [len(y)/2, len(y)/2]  # expected uniform distribution
chi2, p = chisquare(obs, f_exp=exp)
if p < 0.05:
    chi_result = "distribution is significantly different from 50/50."
else:
    chi_result = "no significant difference, cannot conclude imbalance statistically."

print("Chi-square:", round(chi2, 2), "; p-value:", round(p, 6), "; ", chi_result)


# Mutation frequency inspection and summary for overall dataset
mutation_features = X.iloc[:, -20:] # selecting the last 20 columns
mutation_counts = mutation_features.sum().sort_values(ascending=False)
mutation_summary = pd.DataFrame({
    "Mutation": mutation_counts.index,
    "Frequency": mutation_counts.values,
    "Percentage": (mutation_counts.values / len(X) * 100).round(2)
})
print(mutation_summary) 

# Mutation frequency of LGG instances
lgg_index = y[y["Grade"] == 0].index
lgg_mutation_counts = mutation_features.loc[lgg_index].sum().sort_values(ascending=False)
lgg_summary = pd.DataFrame({
    "Mutation": lgg_mutation_counts.index,
    "Frequency": lgg_mutation_counts.values,
    "Percentage": (lgg_mutation_counts.values / len(lgg_index) * 100).round(2)
})
print("LGG Mutation Frequency Summary:")
print(lgg_summary)

# Mutation frequency of GBM instances
gbm_index = y[y["Grade"] == 1].index
gbm_mutation_counts = mutation_features.loc[gbm_index].sum().sort_values(ascending=False)
gbm_summary = pd.DataFrame({
    "Mutation": gbm_mutation_counts.index,
    "Frequency": gbm_mutation_counts.values,
    "Percentage": (gbm_mutation_counts.values / len(gbm_index) * 100).round(2)
})
print("\nGBM Mutation Frequency Summary:")
print(gbm_summary)


# Feature selection using L2 regularization or Ridge regression method
# Standardize features (important for regularization)
# Handle numerical and categorical features separately to run successfully on ridge regression
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
X_numeric = X[numeric_cols]
X_categorical = pd.get_dummies(X[categorical_cols], drop_first=True) # One-hot encode categorical features
X_processed = pd.concat([X_numeric, X_categorical], axis=1) # Combine numeric and encoded categorical features
scaler = StandardScaler()
X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])

# Fit logistic regression with L2 penalty (Ridge)
ridge_clf = LogisticRegression(penalty='l2', solver='liblinear', C=1.0)
ridge_clf.fit(X_processed, y["Grade"])

# Create feature importance summary
feature_importance = pd.DataFrame({
    "Feature": X_processed.columns,
    "Coefficient": ridge_clf.coef_[0],
    "Abs_Coefficient": abs(ridge_clf.coef_[0])
}).sort_values(by="Abs_Coefficient", ascending=False)

print(feature_importance)