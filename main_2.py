# %%
# importing libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functions_1

# %%
# Set path to the current working directory
current_directory = os.getcwd()
data_path = os.path.join(current_directory, 'pointclouds-500')

# List all .xyz files in the directory
file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.xyz')]

# looping over files list to extract xyz values
point_cloud = []
for file in file_paths:
    df = pd.read_csv(file, delimiter=" ", names=['x', 'y', 'z'])
    x_values = df['x'].values
    y_values = df['y'].values
    z_values = df['z'].values
    point_cloud.append(np.column_stack((x_values, y_values, z_values)))

# %%
# visualise elements individually in 3D space (optional)
t = int(input("Object visualization query: "))  # the item no. for 3D viz
x_data = point_cloud[t][:, 0]
y_data = point_cloud[t][:, 1]
z_data = point_cloud[t][:, 2]

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_data, y_data, z_data, c=z_data, alpha=1)
plt.show()


# %%
def pairplot_features(dataframe):
    sns.set(style="ticks")
    g = sns.pairplot(dataframe, hue='label')
    plt.suptitle("Pairplot of Features", y=1.02, fontsize=20)
    plt.legend(loc='upper right')
    plt.show()

# %%
features = functions_1.Features(point_cloud)
df_features = features.df

# %%
pairplot_features(df_features)

# %%
# separate the df_features (X) and labels (y) to train and test the model
X = df_features.drop('label', axis=1)
y = df_features['label']

# %%
# Find out the most important features per classifier
svm_accuracy, svm_mean_accuracy, svm_train_acc = functions_1.SVM_classifier(X,y, 0.2, feat_importance=True)
rf_accuracy, rf_mean_accuracy, rf_train_acc = functions_1.RF_classifier(X, y, 0.2, feat_importance=True)

# %%
# Drop the least important features from the training data
X_updated_svm = df_features.drop(['label', 'density', 'projected_bb'], axis=1)
X_updated_rf = df_features.drop(['label', 'average_height', 'projected_bb'], axis=1)

# %%
# Running the classieifers only with the best 4 features
b_svm_accuracy, b_svm_mean_accuracy, b_svm_train_acc = functions_1.SVM_classifier(X_updated_svm,y, 0.2)
b_rf_accuracy, b_rf_mean_accuracy, b_rf_train_acc = functions_1.RF_classifier(X_updated_rf, y, 0.2)

# %%
# Hyperparameter tuning of SVM method
svm_acc_6f, svm_dict_6f = functions_1.HyperparameterSVM(X, y, 0.2)
svm_acc_4f, svm_dict_4f = functions_1.HyperparameterSVM(X_updated_svm, y, 0.2)

# %%
# barplot of some experiments with SVM method
x_bar_SVM = ['Best features default', 'All features default', 'Best features optimized', 'All features optimized']
y_bar_SVM = [b_svm_accuracy, svm_accuracy, svm_acc_4f, svm_acc_6f]
plt.bar(x_bar_SVM, y_bar_SVM)
plt.ylim(0.9, 1)
plt.xticks(size=10)
plt.title('Experiments and evaluation SVM method')
plt.ylabel('Overall accuracy')

# %%
# Tuning the hyperparameters for Random Forest 
rf_acc_6f, rf_dict_6f = functions_1.HyperparameterRF(X, y, 0.2)
rf_acc_4f, rf_dict_4f = functions_1.HyperparameterRF(X_updated_rf, y, 0.2)

# %%
# barplot of some experiments with RF method
x_bar_RF = ['Best features default', 'All features default', 'Best features optimized', 'All features optimized']
y_bar_RF = [b_rf_accuracy, rf_accuracy, rf_acc_4f, rf_acc_6f]
plt.bar(x_bar_RF, y_bar_RF)
plt.ylim(0.9, 1)
plt.xticks(size=10)
plt.title('Experiments and evaluation RF method')
plt.ylabel('Overall accuracy')

# %%
# Learning curves with all features
functions_1.learning_curve_viz(X, y, 'SVM')
functions_1.learning_curve_viz(X, y, 'RF')

# %%
# Learning curves with the best 4 features
functions_1.learning_curve_viz(X_updated_svm, y, 'SVM')
functions_1.learning_curve_viz(X_updated_rf, y, 'RF')

# %%
# Final results of the classification methods
final_svm_accuracy, final_svm_mean_accuracy, final_svm_train_acc = functions_1.SVM_classifier(X, y, 0.4, kernel=svm_dict_6f['kernel'],
                                                                                            C=svm_dict_6f['C'],
                                                                                             max_iter=svm_dict_6f['max_iter'],
                                                                                              decision_func_shape=svm_dict_6f['decision_func_shape'],
                                                                                               class_weight=svm_dict_6f['class_weight'],
                                                                                                 confusion_mat=True)
final_rf_accuracy, final_rf_mean_accuracy, final_rf_train_acc = functions_1.RF_classifier(X, y, 0.4, n_estimators=rf_dict_6f['n_estimators'],
                                                                                        criterion=rf_dict_6f['criterion'],
                                                                                         max_features=rf_dict_6f['max_features'],
                                                                                          bootstrap=rf_dict_6f['bootstrap'],
                                                                                           max_samples=rf_dict_6f['max_samples'],
                                                                                             confusion_mat=True)
print(f'final overall accuracy of SVM: {final_svm_accuracy} and final overall accuracy of RF: {final_rf_accuracy}')
print(f'final mean per class accuracy of SVM: {final_svm_mean_accuracy} and final mean per class accuracy of RF: {final_rf_mean_accuracy}')

# %%



