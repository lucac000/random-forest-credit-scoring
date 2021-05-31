import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, average_precision_score


df = pd.read_excel('C:/Users/39324/Desktop/API and Big Data/dataset3.xlsx')
# 1 = good, 0 = default
print(df)

# splitting in training, validation and test...
train_predictors, rest_predictors, train_response, rest_response = train_test_split(df.drop('default.payment.next.month', axis=1), df['default.payment.next.month'], train_size=0.7, random_state=0)

validation_predictors, test_predictors, validation_response, test_response = train_test_split(rest_predictors, rest_response, train_size=0.5, random_state=0)


  # model definition + training
n_estimators = 70
max_depth = 70
min_samples_split = 28
min_samples_leaf = 28
min_impurity_decrease = 0
parameters = {"n_estimators": n_estimators,
      "max_depth": max_depth,
      "min_samples_split": min_samples_split,
      "min_samples_leaf": min_samples_leaf,
     "min_impurity_decrease": min_impurity_decrease,
       }
classifier = RandomForestClassifier(**parameters)
classifier.fit(train_predictors, train_response)


# Feature Selection
print(classifier.feature_importances_)
importances = classifier.feature_importances_
indices = np.argsort(importances)
features = train_predictors.columns
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

#pip install shapely
import shap
rf_explainer = shap.TreeExplainer(classifier)
shap_values = rf_explainer.shap_values(train_predictors)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[1], train_predictors)
# When plotting, we call shap_values[1]. For classification problems, 
#    there is a separate array of SHAP values for each possible outcome. 

print(shap_values[0])
print(shap_values[1])


   # model definition + training
for i in range(10):
 
    n_estimators = 10*(i+1)
    max_depth = 10*(i+1)
    min_samples_split = 4*(i+1)
    min_samples_leaf = 4*(i+1)
    min_impurity_decrease = 0
    parameters = {"n_estimators": n_estimators,
      "max_depth": max_depth,
      "min_samples_split": min_samples_split,
      "min_samples_leaf": min_samples_leaf,
      "min_impurity_decrease": min_impurity_decrease,
      "random_state": 0,
       }
    classifier = RandomForestClassifier(**parameters)
    classifier.fit(train_predictors, train_response)
    # Compute the ROC curve and AUC
    y_test = validation_response
    Q = classifier.predict_proba(validation_predictors)[:,1]
    fpr, tpr, _ = roc_curve(y_test, Q)
    roc_auc = auc(fpr,tpr)
    print('hyperparameters set %d - AUC = %0.4f' % (i, roc_auc))


# Measure overall quality of our classifier


    i=7
n_estimators = 10*(i+1)
max_depth = 10*(i+1)
min_samples_split = 4*(i+1)
min_samples_leaf = 4*(i+1)
min_impurity_decrease = 0
parameters = {"n_estimators": n_estimators,
      "max_depth": max_depth,
      "min_samples_split": min_samples_split,
      "min_samples_leaf": min_samples_leaf,
      "min_impurity_decrease": min_impurity_decrease,
      "random_state": 0,
       }
classifier = RandomForestClassifier(**parameters)
classifier.fit(train_predictors, train_response)


y_test = test_response
y_pred = classifier.predict(test_predictors)


THRESHOLD = [.5, .75, .80, .85]
results = pd.DataFrame(columns=["THRESHOLD", "accuracy", "recall", "tnr", "fpr", "precision", "f1_score"]) # df to store results
results['THRESHOLD'] = THRESHOLD                                                                           # threshold column
n_test = len(y_test)
Q = classifier.predict_proba(test_predictors)[:,1]
j = 0                                                                                                      
for i in THRESHOLD:                                                                                        # iterate over each threshold        
                                                                         # fit data to model
    preds = np.where(Q>i, 1, 0)                                       # if prob > threshold, predict 1
    
    cm = (confusion_matrix(y_test, preds,labels=[1, 0], sample_weight=None)/n_test)*100 
    # confusion matrix (in percentage)
    
    print('Confusion matrix for threshold =',i)
    print(cm)
    print(' ')      
    
    TP = cm[0][0]                                                                                          # True Positives
    FN = cm[0][1]                                                                                          # False Positives
    FP = cm[1][0]                                                                                          # True Negatives
    TN = cm[1][1]                                                                                          # False Negatives
        
    results.iloc[j,1] = accuracy_score(y_test, preds) 
    results.iloc[j,2] = recall_score(y_test, preds)
    results.iloc[j,3] = TN/(FP+TN)                                                                         # True negative rate
    results.iloc[j,4] = FP/(FP+TN)                                                                         # False positive rate
    results.iloc[j,5] = precision_score(y_test, preds)
    results.iloc[j,6] = f1_score(y_test, preds)
   
    j += 1

print('ALL METRICS')
print(results.T.to_string(header=False))



plt.figure(figsize=(8,6))      # format the plot size
lw = 1.5
plt.plot(fpr, tpr, color='darkorange', marker='.',
         lw=lw, label='Random Forest (AUC = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
         label='Random Prediction (AUC = 0.5)' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()
