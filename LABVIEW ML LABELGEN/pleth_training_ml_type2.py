#!/usr/bin/env python
# coding: utf-8

# # Final Thesis Project - Training Data

# This Notebook tests out different ML models and check the scores. 
# 
# The training dataset contains a total of ? samples. 

# In[6]:


import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')


# In[6]:


import os
print(os.getcwd())


# In[7]:


# Loading Data
data_train = np.load('data_train_type2.npy', allow_pickle=True)
labels_train = np.load('labels_train_type2.npy', allow_pickle=True)

print(data_train.shape, labels_train.shape)


# In[8]:


'''
# Loading Data
napoli_test = np.load('napoli_data.npy', allow_pickle=True)
labels_test = np.load('napoli_labels.npy', allow_pickle=True)

#print(da_train.shape, labels_train.shape)
'''


# In[9]:


# Labels Encoding

labels_names = []


# In[10]:


# Counting number samples per class
vals, counts = np.unique(labels_train, return_counts=True)

plt.bar(vals, counts)
plt.xticks(range(7),range(7))
plt.xlabel('Classes',size=20)
plt.ylabel('# Samples per Class', size=20)
plt.title('Training Data (Total = '+str(data_train.shape[0])+' samples)',size=16);


# In[11]:


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from skimage.transform import resize
from sklearn.svm import SVC
#from sklearn.metrics import roc_curve, roc_auc_score
#import cv2


# In[12]:


X_train, X_test, t_train, t_test = train_test_split(data_train, labels_train, 
                                                   test_size=0.2,
                                                   stratify=labels_train,
                                                   random_state=0)
print(X_train.shape)
print(t_train.shape)
print(X_test.shape)
print(t_test.shape)


# In[13]:


'''
X_napoli_train, X_napoli_test, t_napoli_train, t_napoli_test = train_test_split(napoli_test, labels_test, 
                                                   test_size=0.5,
                                                   random_state=0)
print(X_train.shape)
print(t_train.shape)
print(X_test.shape)
print(t_test.shape)
'''


# ---

# ## 1.) LDA + LOGISTIC REGRESSION (Model No.1)

# In[73]:


mod1 = Pipeline([('SCALER', StandardScaler()),
                 ('LDA', LDA(n_components=15)),
                 ('LOGRES', LogisticRegression())])
mod1.fit(X_train, t_train)


# In[74]:


######## GRIDSEARCH CROSS-VALIDATION ##########

# Parameter grid
param_grid = {
    'LDA__n_components': [2, 3, 4, 5, 6],  # Test different number of components for LDA
    'LOGRES__penalty': ['l1', 'l2'],  # Test different regularization penalties
    'LOGRES__C': [0.1, 1.0, 10.0, 100.0, 1000.0],  # Test different values of regularization strength
    'LOGRES__solver': ['liblinear', 'saga']  # Test different solvers
}

grid_search = GridSearchCV(mod1, param_grid, cv=5, scoring='accuracy')

# Perform grid search, fit on data
grid_search.fit(X_train, t_train)

# Get the best parameters found
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use best model found
lda_best_model = grid_search.best_estimator_


# In[75]:


#pred_test1 = mod1.predict(X_napoli_test)


# In[76]:


# Make predictions on test set
pred_test1 = lda_best_model.predict(X_test)

print('LDA + LR:')
print('Training Accuracy: \n ',accuracy_score(t_test, pred_test1))
print ('F1_score:\n',f1_score(t_test, pred_test1, average=None))
print('Confusion matrix:')
print(confusion_matrix(t_test, pred_test1))


# In[77]:


'''
print('LR\n')
print('Accuracy:\n',accuracy_score(t_napoli_test, pred_test1))
print ('F1_score:\n',f1_score(t_napoli_test, pred_test1, average=None))
print('Confusion matrix:\n',confusion_matrix(t_napoli_test, pred_test1))
'''


# ## 2.) PCA + LOGISTIC REGRESSION (Model No. 2)

# In[78]:


N, D = np.shape(X_train)
pca = PCA(n_components=min(N,D))
pca.fit(X_train)

plt.step(range(1,min(N,D)+1),np.cumsum(pca.explained_variance_ratio_)*100)

print(np.where(np.cumsum(pca.explained_variance_ratio_)>=0.9))
print(np.cumsum(pca.explained_variance_ratio_)[15])
plt.xlabel('Number of principal components');
plt.ylabel('% Variance explained');


# In[79]:


mod2 = Pipeline([('SCALER', StandardScaler()),
                 ('PCA', PCA(n_components=15)),
                 ('LOGREG', LogisticRegression(random_state=0, tol=0.01))]) 
mod2.fit(X_train, t_train)


# In[80]:


######## GRIDSEARCH CROSS-VALIDATION ##########

# Parameter grid
param_grid = {
    'PCA__n_components': [10, 12, 15, 17, 19],  # Test different number of components for PCA
    'LOGREG__penalty': ['l1', 'l2'],  # Test different regularization penalties
    'LOGREG__C': [0.1, 1.0, 10.0, 100.0, 1000.0],  # Test different values of regularization strength
}

# Create a pipeline with standard scaler, PCA, and Logistic Regression
mod2 = Pipeline([
    ('SCALER', StandardScaler()),
    ('PCA', PCA(n_components=16)),
    ('LOGREG', LogisticRegression(random_state=0, tol=0.01))
])

grid_search = GridSearchCV(mod2, param_grid, cv=5, scoring='accuracy')

# Perform grid search, fit on data
grid_search.fit(X_train, t_train)

# Get the best parameters found
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model found
pca_best_model = grid_search.best_estimator_


# In[81]:


# Make predictions on test set
pred_test2 = pca_best_model.predict(X_test)

print('PCA + LR:')
print('Training Accuracy: \n ',accuracy_score(t_test, pred_test2))
print ('F1_score:\n',f1_score(t_test, pred_test2, average=None))
print('Confusion matrix:')
print(confusion_matrix(t_test, pred_test2))


# In[82]:


'''
pred_test2 = mod2.predict(X_napoli_test)

print('Accuracy:\n',accuracy_score(t_napoli_test, pred_test2))
print ('F1_score:\n',f1_score(t_napoli_test, pred_test2, average=None))
print('Confusion matrix:\n',confusion_matrix(t_napoli_test, pred_test2))
'''


# ## 3.) Random Forest (Model No. 3)

# In[83]:


from sklearn.ensemble import RandomForestClassifier


# In[84]:


# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)


# In[85]:


######## GRIDSEARCH CROSS-VALIDATION ##########

# Parameter grid
param_grid = {
    'n_estimators': [50, 150, 250],  # Number of trees in the forest
    #'max_depth': [None, 20, 40],  # Maximum depth of the trees
    #'min_samples_split': [2, 10, 20],  # Minimum number of samples required to split an internal node
    #'min_samples_leaf': [1, 4, 10],  # Minimum number of samples required to be at a leaf node
}

grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')

# Perform grid search, fit on data
grid_search.fit(X_train, t_train)

# Get the best parameters found
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model found
rf_best_model = grid_search.best_estimator_


# In[86]:


# Make predictions on test set
pred_test3 = rf_best_model.predict(X_test)

print('RF:')
print('Training Accuracy: \n ',accuracy_score(t_test, pred_test3))
print ('F1_score:\n',f1_score(t_test, pred_test3, average=None))
print('Confusion matrix:')
print(confusion_matrix(t_test, pred_test3))


# In[87]:


'''
pred_test3 = rf_classifier.predict(X_napoli_test)

print('With Random Forest:')
print('Test Accuracy Score = ',accuracy_score(t_napoli_test, pred_test3))
print('Confusion matrix:')
print(confusion_matrix(t_napoli_test, pred_test3))
'''


# ## 4.) XGBoost (Model No.4)

# In[14]:


import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_tree
import seaborn as sns


# In[15]:


# Create an XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=5, random_state=0)


# In[16]:


######## GRIDSEARCH CROSS-VALIDATION ##########

# Parameter grid
param_grid = {
    'n_estimators': [50, 150, 250],  # Number of trees
    #'max_depth': [3, 5, 7],  # Maximum depth of a tree
    #'learning_rate': [0.01, 0.2, 0.4],  # Learning rate
    #'gamma': [0, 0.2, 0.4],  # Minimum loss reduction required to make a further partition
    #'subsample': [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
}

grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5, scoring='accuracy')

# Perform grid search, fit on data
grid_search.fit(X_train, t_train)

# Get the best parameters found
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model found
xgb_best_model = grid_search.best_estimator_


# ## Feature Importance for XGB model

# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Get feature importance scores from the trained XGBoost model
feature_importance = xgb_best_model.feature_importances_

# Replace 'column_names' with the actual variable name containing the column names
#column_names = ['Ti (msec)', 'PIF', 'PEF', 'EV', 'RT (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'RH', 'Tbox', 'VCF', 'AV', 'Sr']
#column_names = ['Ti (msec)', 'Te (msec)', 'PIF', 'PEF', 'TV', 'EV', 'RT (msec)', 'MV', 'P (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'RH', 'Sr', 'Phase']
column_names = ['Ti (msec)', 'Te (msec)', 'PIF', 'PEF', 'TV', 'EV', 'RT (msec)', 'MV', 'P (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'Sr', 'Phase']

# Create a DataFrame to hold feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': column_names, 'Importance': feature_importance})

# Sort the DataFrame by importance score in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top features
print(feature_importance_df)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()


# In[18]:


get_ipython().run_line_magic('matplotlib', 'qt')
plot_tree(xgb_best_model, num_trees=25, rankdir='LR')  # Change num_trees to visualize a different tree if needed
plt.show()

booster = xgb_best_model.get_booster()
booster.dump_model('xgb_best_model.txt')


# ## K-fold cross-validation (Used to estimate the skill of the model on new data)

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Perform k-fold cross-validation
cv_scores = cross_val_score(xgb_best_model, X_train, t_train, cv=5)

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Visualize cross-validation scores
plt.figure(figsize=(10, 1))
sns.boxplot(x=cv_scores)
plt.title('Cross-Validation Scores')
plt.xlabel('Scores')
plt.ylabel('CV Fold')
plt.show()


# In[20]:


# Make predictions on test set
pred_test4 = xgb_best_model.predict(X_test)

print('XGB:')
print('Training Accuracy: \n ',accuracy_score(t_test, pred_test4))
print ('F1_score:\n',f1_score(t_test, pred_test4, average=None))
print('Confusion matrix:')
print(confusion_matrix(t_test, pred_test4))


# ## 5.) CNN (Model No. 5)

# In[95]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


X_train_full = np.load('data_train.npy').T
t_train_full = np.load('labels_train_corrected.npy')

X_train_full.shape, t_train_full.shape


# In[ ]:


from sklearn.model_selection import train_test_split

# Training and Test sets
X_training, X_test, t_training, t_test = train_test_split(X_train_full, 
                                                  t_train_full, 
                                                  shuffle=True,
                                                  stratify=t_train_full,
                                                  test_size=0.15)
# Train and validation sets
X_train, X_val, t_train, t_val = train_test_split(X_training, 
                                                  t_training, 
                                                  shuffle=True,
                                                  stratify=t_training,
                                                  test_size=0.2)

X_training.shape, t_training.shape, X_train.shape, t_train.shape, X_val.shape, t_val.shape


# In[ ]:


del X_train_full, t_train_full
# free up space


# In[ ]:



X_training = X_training.reshape(X_training.shape[0], 300, 300, 3)/255.0

X_train = X_train.reshape(X_train.shape[0], 300, 300, 3)/255.0

X_val = X_val.reshape(X_val.shape[0], 300, 300, 3)/255.0

X_test = X_test.reshape(X_test.shape[0], 300, 300, 3)/255.0


# In[ ]:


model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation='relu', padding='same', input_shape=[300,300,3]), 
    keras.layers.MaxPooling2D(2), 
    keras.layers.Conv2D(128, 3, activation='relu', padding='same'), 
    keras.layers.MaxPooling2D(2),
    keras.layers.MaxPooling2D(2), 
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
             optimizer=keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999),
             metrics=['accuracy'])


# In[ ]:


model.fit(X_train, t_train, epochs=2, batch_size=32,
          validation_data=(X_val, t_val),
         callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])


# In[ ]:


model.evaluate(X_test, t_test)


# # Exporting Best Performing ML Model

# In[21]:


import joblib

# Save best model pkl file
joblib.dump(xgb_best_model, 'best_pleth_ml_model_type2.pkl')


# # Testing on Unseen Data

# In[ ]:


# Loading Unseen Data
data_unseen_type2 = np.load('data_unseen_type2.npy', allow_pickle=True)
labels_unseen_type2 = np.load('labels_unseen_type2.npy', allow_pickle=True)

print(data_unseen_type2.shape, labels_unseen_type2.shape)


# In[ ]:


# Counting number samples per class
vals, counts = np.unique(labels_unseen_type2, return_counts=True)

plt.bar(vals, counts)
plt.xticks(range(7),range(7))
plt.xlabel('Classes',size=20)
plt.ylabel('# Samples per Class', size=20)
plt.title('Training Data (Total = '+str(data_unseen_type2.shape[0])+' samples)',size=15);


# In[ ]:


# Load the model from the file
loaded_model_type2 = joblib.load('best_pleth_ml_model_type2.pkl')

# Make predictions on unseen data
pred_unseen_type2 = loaded_model_type2.predict(data_unseen_type2)

print('XGB:')
print('Testing Accuracy: \n ',accuracy_score(labels_unseen_type2, pred_unseen_type2))
print ('F1_score:\n',f1_score(labels_unseen_type2, pred_unseen_type2, average=None))
print('Confusion matrix:')
print(confusion_matrix(labels_unseen_type2, pred_unseen_type2))


# ## Label Generator 

# In[26]:


'''
#Path for excel file to generate labels for
new_path_type2 = r"C:\Users\edward.luca\Github\THC_Rat_analyis_ML\thc_data\day1\baseline\day1_baseline_control_3.25.24.rf_1.iox_clean.xlsx"

#Read excel file
new_df_type2 = pd.read_excel(new_path_type2)

# Full column headers
full_column_headers = ['Sample', 'CPU Date', 'CPU Time', 'Site Time', 'Period Time', 'Protocol Type', 'Storage ID', 'First Beat ID', 'Last Beat ID', 'Ti (msec)', 'Te (msec)', 'PIF', 'PEF', 'TV', 'EV', 'RT (msec)', 'MV', 'P (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'RH', 'Tbox', 'Tbody', 'Sr', 'Phase']

# Get ML column headers
column_headers = ['Ti (msec)', 'Te (msec)', 'PIF', 'PEF', 'TV', 'EV', 'RT (msec)', 'MV', 'P (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'Sr', 'Phase']

#Extract features
new_features_type2 = new_df_type2[['Ti (msec)', 'Te (msec)', 'PIF', 'PEF', 'TV', 'EV', 'RT (msec)', 'MV', 'P (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'Sr', 'Phase']]

#Convert to numpy file (maybe)
np.save('new_data_train_type2.npy', new_features_type2)

# Load the saved numpy file
new_data_train_type2 = np.load('new_data_train_type2.npy', allow_pickle=True)

# Make predictions on saved numpy file
# Load the model from the file
loaded_model_type2 = joblib.load('best_pleth_ml_model_type2.pkl')
pred_type2 = loaded_model_type2.predict(new_data_train_type2)

# Create new column to store the generated labels and add column name to column_headers list
new_column = np.ones((new_data_train_type2.shape[0], 1))
full_column_headers += ["Generated Labels"]

# Append the new column
modified_data = np.concatenate((new_data_train_type2, new_column), axis=1)

# Fill the new column with generated labels
modified_data[:,-1] = pred_type2

# Load NumPy array
generated_label = np.load('generatelabeltest.npy')

# Get the generated labels
generated_labels = generated_label[:, -1]

# Count the occurrences of each label
label_counts = np.unique(generated_labels, return_counts=True)

# Define label descriptions
label_descriptions = {
    0: "Normal / Quiet Breath",
    1: "Sigh breath",
    2: "Sniffing breath",
    3: "Random Apnea",
    4: "Type II Apnea"
}

# Print the counts with descriptions
for label, count in zip(label_counts[0], label_counts[1]):
    description = label_descriptions[int(label)]
    print(f"{description} count ({int(label)}): {count}")
    
# Convert to pandas DataFrame
df = pd.DataFrame(generated_label, columns=full_column_headers)
# Convert and export DataFrame to Excel
excel_file_path = r"C:\Users\edward.luca\Github\THC_Rat_analyis_ML\thc_data\day1\baseline\day1_baseline_control_3.25.24.rf_1.iox_labelgen.xlsx"
df.to_excel(excel_file_path, index=False)
'''


# In[23]:


#Path for excel file to generate labels for
new_path_type2 = r"C:\Users\edward.luca\Github\THC_Rat_analyis_ML\thc_data\day3\chemo\day3_chemo_thc_3.27.24.rf_1.iox_clean2.xlsx"

#Read excel file
new_df_type2 = pd.read_excel(new_path_type2)

# Full column headers
full_column_headers = ['Sample', 'CPU Date', 'CPU Time', 'Site Time', 'Period Time', 'Protocol Type', 'Storage ID', 'First Beat ID', 'Last Beat ID', 'Ti (msec)', 'Te (msec)', 'PIF', 'PEF', 'TV', 'EV', 'RT (msec)', 'MV', 'P (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'RH', 'Tbox', 'Tbody', 'Sr', 'Phase']

# Get ML column headers
column_headers = ['Ti (msec)', 'Te (msec)', 'PIF', 'PEF', 'TV', 'EV', 'RT (msec)', 'MV', 'P (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'Sr', 'Phase']

#Extract features
new_features_type2 = new_df_type2[['Ti (msec)', 'Te (msec)', 'PIF', 'PEF', 'TV', 'EV', 'RT (msec)', 'MV', 'P (msec)', 'f (bpm)', 'EIP (msec)', 'EEP (msec)', 'Penh', 'EF50', 'Sr', 'Phase']]

#Convert to numpy file (maybe)
np.save('new_data_train_type2.npy', new_features_type2)

# Load the saved numpy file
new_data_train_type2 = np.load('new_data_train_type2.npy', allow_pickle=True)

# Make predictions on saved numpy file
# Load the model from the file
loaded_model_type2 = joblib.load('best_pleth_ml_model_type2.pkl')
pred_type2 = loaded_model_type2.predict(new_data_train_type2)

# Create new column to store the generated labels and add column name to column_headers list
new_column = np.ones((new_data_train_type2.shape[0], 1))
full_column_headers += ["Generated Labels"]

# Append the new column
modified_data = np.concatenate((new_data_train_type2, new_column), axis=1)

# Fill the new column with generated labels
modified_data[:,-1] = pred_type2

# Create a new column to store the generated labels
new_df_type2["Generated Labels"] = pred_type2

# Save predicted labels to 'generatelabeltest.npy' file
np.save('generatelabeltest.npy', pred_type2)

# Load the generated labels
generated_labels = np.load('generatelabeltest.npy')

# Define label descriptions
label_descriptions = {
    0: "Normal / Quiet Breath",
    1: "Sigh breath",
    2: "Sniffing breath",
    3: "Random Apnea",
    4: "Type II Apnea"
}

# Count the occurrences of each label
label_counts = np.unique(generated_labels, return_counts=True)

# Initialize total breath count
total_breath_count = 0

# Print the counts with descriptions
for label, count in zip(label_counts[0], label_counts[1]):
    description = label_descriptions[int(label)]
    print(f"{description} count ({int(label)}): {count}")
    # Increment total breath count
    total_breath_count += count

# Print total breath count
print(f"Total breath count: {total_breath_count}")

# Add a new column for generated labels to the original DataFrame
new_df_type2["Generated Labels"] = generated_labels

# Convert and export DataFrame to Excel with all columns
excel_file_path = r"C:\Users\edward.luca\Github\THC_Rat_analyis_ML\thc_data\day3\chemo\day3_chemo_thc_3.27.24.rf_1.iox_labelgen2.xlsx"
new_df_type2.to_excel(excel_file_path, index=False)


# In[ ]:




