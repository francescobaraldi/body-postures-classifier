import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, \
    precision_score, \
    recall_score, \
    f1_score, \
    precision_recall_curve, \
    roc_curve

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 20)

df = pd.read_csv('datasets/dataset-har-PUC-Rio-ugulino.csv', sep=';')


def kdeplot(feature, labels=list(df['class'].unique())):
    plt.figure(figsize=(10, 6))
    plt.title("KDE for {}".format(feature))
    for label in labels:
        ax = sns.kdeplot(df[df['class'] == label][feature].dropna(), label='Position: {}'.format(label))
        ax.legend()


def barplot_percentages(feature, orient='v', axis_name="percentage of subjects"):
    g = df.groupby(feature)["class"].value_counts().to_frame()
    g = g.rename({"class": axis_name}, axis=1).reset_index()
    g[axis_name] = g[axis_name] / len(df)
    if orient == 'v':
        ax = sns.barplot(x=feature, y=axis_name, hue='class', data=g, orient=orient)
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])
    else:
        ax = sns.barplot(x=axis_name, y=feature, hue='class', data=g, orient=orient)
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])
    ax.plot()


def execute_scores(model_name, y_test, y_pred):
    print("Metriche per l'algoritmo", model_name)
    print('Accuracy is ', accuracy_score(y_test, y_pred))
    print('Precision is ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall is ', recall_score(y_test, y_pred, average='weighted'))
    print('F1-Score is ', f1_score(y_test, y_pred, average='weighted'))
    # print('AUC is ', roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr'))


# EDA
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())

df['body_mass_index'] = df['body_mass_index'].str.replace(',', '.')
df['body_mass_index'] = df['body_mass_index'].astype('float64')
df['how_tall_in_meters'] = df['how_tall_in_meters'].str.replace(',', '.')
df['how_tall_in_meters'] = df['how_tall_in_meters'].astype('float64')
df['z4'] = pd.to_numeric(df['z4'], errors='coerce')
df = df.drop(df[df['z4'].isna()].index)  # Elimino la riga con z4 NaN, alternativa è cambiare il valore
df['z4'] = df['z4'].astype('int64')
df['gender'], _ = pd.factorize(df['gender'])
df['class'], _ = pd.factorize(df['class'])

X = df.drop(['user', 'class'], axis=1)
y = df['class'].values
X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(X, y, test_size=0.2)

########################################################################################################################
# CROSS-VALIDATION PER MODEL SELECTION VERSIONE 1#
########################################################################################################################
# DECISION TREE
decision_tree_v1 = tree.DecisionTreeClassifier(random_state=3)
iperparametri = {'max_depth': list(range(1, 20)), 'class_weight': ['balanced', None], 'criterion': ['gini', 'entropy'],
                 'splitter': ['best', 'random']}
clf = GridSearchCV(estimator=decision_tree_v1, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v1, y_train_v1)
print('Decision Tree')
print('The best value for parameter max_depth is', clf.best_params_.get('max_depth'),
      'the best value for parameter class_weight is', clf.best_params_.get('class_weight'),
      'the best value for parameter criterion is', clf.best_params_.get('criterion'),
      'the best value for parameter splitter is', clf.best_params_.get('splitter'),
      'since it leads to F1-score =', clf.best_score_)
decision_tree_definitivo_v1 = tree.DecisionTreeClassifier(random_state=3, max_depth=18, class_weight=None,
                                                          criterion='entropy', splitter='best')  # F1-score = 0.9847166535002444
decision_tree_definitivo_v1.fit(X_train_v1, y_train_v1)
y_pred = decision_tree_definitivo_v1.predict(X_test_v1)
execute_scores("Decision Tree v1", y_test_v1, y_pred)

mat = confusion_matrix(y_test_v1, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Valori reali')
plt.ylabel('Valori predetti')
plt.show()

# SVM
svm_v1 = SVC(random_state=3, verbose=True)
iperparametri = {'C': [1e-1, 1e3], 'kernel': ['linear', 'rbf'], 'class_weight': ['balanced', None],
                 'max_iter': [100, 200], 'decision_function_shape': ['ovo', 'ovr']}
clf = GridSearchCV(estimator=svm_v1, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v1, y_train_v1)
print('The best value for parameter C is', clf.best_params_.get('C'),
      'the best value for parameter kernel is', clf.best_params_.get('kernel'),
      'the best value for parameter class_weight is', clf.best_params_.get('class_weight'),
      'the best value for parameter max_iter is', clf.best_params_.get('max_iter'),
      'the best value for parameter decision_function_shape is', clf.best_params_.get('decision_function_shape'),
      'since these lead to F1-score =', clf.best_score_)
svm_definitivo_v1 = SVC(random_state=3, C=1e-1, kernel='linear', class_weight='balanced', max_iter=100,
                        decision_function_shape='ovo')  # F1-score = 0.41647451327009444
svm_definitivo_v1.fit(X_train_v1, y_train_v1)
y_pred = svm_definitivo_v1.predict(X_test_v1)
execute_scores("SVM v1", y_test_v1, y_pred)

mat = confusion_matrix(y_test_v1, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Valori reali')
plt.ylabel('Valori predetti')
plt.show()

# NEURAL NETWORK
nn_v1 = MLPClassifier(random_state=3, verbose=True, tol=1e-6)
iperparametri = {'hidden_layer_sizes': [(100, 100), (50, 50, 50, 50)], 'max_iter': [100, 200],
                 'early_stopping': [True, False], 'activation': ['logistic', 'tanh']}
clf = GridSearchCV(estimator=nn_v1, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v1, y_train_v1)
print('The best value for parameter hidden_layer_sizes is', clf.best_params_.get('hidden_layer_sizes'),
      'the best value for parameter max_iter is', clf.best_params_.get('max_iter'),
      'the best value for parameter early_stopping is', clf.best_params_.get('early_stopping'),
      'the best value for parameter activation is', clf.best_params_.get('activation'),
      'since these lead to F1-score =', clf.best_score_)
nn_definitivo_v1 = MLPClassifier(random_state=3, tol=1e-6, hidden_layer_sizes=(100, 100), max_iter=100,
                                 early_stopping=False, activation='logistic')  # F1-score = 0.985922839366024
nn_definitivo_v1.fit(X_train_v1, y_train_v1)
y_pred = nn_definitivo_v1.predict(X_test_v1)
execute_scores("NN v1", y_test_v1, y_pred)

mat = confusion_matrix(y_test_v1, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Valori reali')
plt.ylabel('Valori predetti')
plt.show()

########################################################################################################################
# CROSS-VALIDATION PER MODEL ASSESSMENT VERSIONE 1#
########################################################################################################################
# DECISION TREE
scores = cross_validate(decision_tree_definitivo_v1, X_train_v1, y_train_v1, cv=5,
                        scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of Decision Tree v1 is', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of Decision Tree v1 is ', np.mean(scores['test_balanced_accuracy']))

# SVM
scores = cross_validate(svm_definitivo_v1, X_train_v1, y_train_v1, cv=5, scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of SVM v1 is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of SVM v1 is ', np.mean(scores['test_balanced_accuracy']))

# NEURAL NETWORK
scores = cross_validate(nn_definitivo_v1, X_train_v1, y_train_v1, cv=5, scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of NN v1 is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of NN v1 is ', np.mean(scores['test_balanced_accuracy']))


# Preprocessing
sns.catplot(y="class", kind="count", data=df, height=2.6, aspect=2.5, orient='h')  # dataset sbilanciato
sns.catplot(y="gender", kind="count", data=df, height=2.6, aspect=2.5, orient='h')
sns.catplot(y="user", kind="count", data=df, height=2.6, aspect=2.5, orient='h')

positional = list(set(df.columns) - {'user', 'gender', 'age', 'how_tall_in_meters', 'weight', 'body_mass_index',
                                     'class'})
for feature in positional:
    kdeplot(feature)

barplot_percentages("user")
barplot_percentages("age")
barplot_percentages("gender")
barplot_percentages("how_tall_in_meters")
barplot_percentages("weight")
barplot_percentages("body_mass_index")
barplot_percentages("x1")

sns.barplot(x="age", y=df['user'], hue='user', data=df, orient='v')
feat = ["user", "gender", "age", "how_tall_in_meters", "weight", "body_mass_index"]

positional = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4']
g = sns.PairGrid(df, y_vars=["z1", "z2", "z3"], x_vars=positional, height=2, hue="class", aspect=1.1)
ax = g.map(plt.scatter, alpha=0.6)

# Matrice di correlazione
corr_matrix = df.corr()
sns.heatmap(corr_matrix)

features = ['how_tall_in_meters', 'x1', 'y1', 'z1', 'y2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4']
X = df[features]
y = df['class'].values

X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(X, y, test_size=0.2)

mms = MinMaxScaler()
mms.fit(X_train_v2)
X_train_v2_scaled = mms.transform(X_train_v2)
X_test_v2_scaled = mms.transform(X_test_v2)

########################################################################################################################
# CROSS-VALIDATION PER MODEL SELECTION VERSIONE 2 #
########################################################################################################################
# DECISION TREE
decision_tree_v2 = tree.DecisionTreeClassifier(random_state=3)
iperparametri = {'max_depth': list(range(1, 20)), 'class_weight': ['balanced', None], 'criterion': ['gini', 'entropy'],
                 'splitter': ['best', 'random']}
clf = GridSearchCV(estimator=decision_tree_v2, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v2_scaled, y_train_v2)
print('Decision Tree v2')
print('The best value for parameter max_depth is', clf.best_params_.get('max_depth'),
      'the best value for class_weight is', clf.best_params_.get('class_weight'),
      'the best value for criterion is', clf.best_params_.get('criterion'),
      'the best value for splitter is', clf.best_params_.get('splitter'),
      'since it leads to balanced F1-score =', clf.best_score_)
decision_tree_definitivo_v2 = tree.DecisionTreeClassifier(random_state=3, max_depth=19, class_weight='balanced',
                                                          criterion='entropy', splitter='best')  # F1-score = 0.9816888361946873
decision_tree_definitivo_v2.fit(X_train_v2_scaled, y_train_v2)
y_pred = decision_tree_definitivo_v2.predict(X_test_v2_scaled)
execute_scores("Decision Tree v2", y_test_v2, y_pred)

mat = confusion_matrix(y_test_v2, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Valori reali')
plt.ylabel('Valori predetti')
plt.show()

# SVM
svm_v2 = SVC(random_state=3, verbose=True)
iperparametri = {'C': [1e-1, 1e3], 'kernel': ['linear', 'rbf'], 'class_weight': ['balanced', None],
                 'max_iter': [100, 200], 'decision_function_shape': ['ovo', 'ovr']}
clf = GridSearchCV(estimator=svm_v2, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v2_scaled, y_train_v2)
print('The best value for parameter C is', clf.best_params_.get('C'),
      'the best value for parameter kernel is', clf.best_params_.get('kernel'),
      'the best value for parameter class_weight is', clf.best_params_.get('class_weight'),
      'the best value for parameter max_iter is', clf.best_params_.get('max_iter'),
      'the best value for parameter decision_function_shape is', clf.best_params_.get('decision_function_shape'),
      'since these lead to F1-score =', clf.best_score_)
svm_definitivo_v2 = SVC(random_state=3, C=1e3, kernel='rbf', class_weight=None, max_iter=200,
                        decision_function_shape='ovo')  # F1-score = 0.4745064536372666
svm_definitivo_v2.fit(X_train_v2_scaled, y_train_v2)
y_pred = svm_definitivo_v2.predict(X_test_v2_scaled)
execute_scores("SVM v2", y_test_v2, y_pred)

mat = confusion_matrix(y_test_v2, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Valori reali')
plt.ylabel('Valori predetti')
plt.show()

# NEURAL NETWORK
nn_v2 = MLPClassifier(random_state=3, verbose=True, tol=1e-6)
iperparametri = {'hidden_layer_sizes': [(100, 100), (50, 50, 50, 50)], 'max_iter': [100, 200],
                 'early_stopping': [True, False], 'activation': ['logistic', 'tanh']}
clf = GridSearchCV(estimator=nn_v2, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v2_scaled, y_train_v2)
print('The best value for parameter hidden_layer_sizes is', clf.best_params_.get('hidden_layer_sizes'),
      'the best value for parameter max_iter is', clf.best_params_.get('max_iter'),
      'the best value for parameter early_stopping is', clf.best_params_.get('early_stopping'),
      'the best value for parameter activation is', clf.best_params_.get('activation'),
      'since these lead to F1-score =', clf.best_score_)
nn_definitivo_v2 = MLPClassifier(random_state=3, tol=1e-6, hidden_layer_sizes=(100, 100), max_iter=200,
                                 early_stopping=False, activation='tanh')  # F1-score = 0.987283880930265
nn_definitivo_v2.fit(X_train_v2_scaled, y_train_v2)
y_pred = nn_definitivo_v2.predict(X_test_v2_scaled)
execute_scores("NN v2", y_test_v2, y_pred)

mat = confusion_matrix(y_test_v2, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Valori reali')
plt.ylabel('Valori predetti')
plt.show()

########################################################################################################################
# CROSS-VALIDATION PER MODEL ASSESSMENT VERSIONE 2 #
########################################################################################################################
# DECISION TREE
scores = cross_validate(decision_tree_definitivo_v2, X_train_v2_scaled, y_train_v2, cv=5,
                        scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of Decision Tree v2 is', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of Decision Tree v2 is ', np.mean(scores['test_balanced_accuracy']))

# SVM
scores = cross_validate(svm_definitivo_v2, X_train_v2_scaled, y_train_v2, cv=5, scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of SVM v2 is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of SVM v2 is ', np.mean(scores['test_balanced_accuracy']))

# NEURAL NETWORK
scores = cross_validate(nn_definitivo_v2, X_train_v2_scaled, y_train_v2, cv=5, scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of NN v2 is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of NN v2 is ', np.mean(scores['test_balanced_accuracy']))

########################################################################################################################
# CROSS-VALIDATION PER MODEL SELECTION VERSIONE 3 #
########################################################################################################################
# Decision Tree - versione migliore è la 1
decision_tree_v3 = RandomForestClassifier(random_state=3, max_depth=18, class_weight=None, criterion='entropy')
iperparametri = {'n_estimators': list(range(1, 20))}
clf = GridSearchCV(estimator=decision_tree_v3, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v1, y_train_v1)
print('Decision Tree v3')
print('The best value for parameter n_estimators is', clf.best_params_.get('n_estimators'),
      'since it leads to balanced F1-score =', clf.best_score_)
decision_tree_definitivo_v3 = RandomForestClassifier(random_state=3, max_depth=18, class_weight=None,
                                                     criterion='entropy', n_estimators=18)  # F1-score = 0.9951017788704887
decision_tree_definitivo_v3.fit(X_train_v1, y_train_v1)
y_pred = decision_tree_definitivo_v3.predict(X_test_v1)
execute_scores("Decision Tree v3", y_test_v1, y_pred)

mat = confusion_matrix(y_test_v1, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Valori reali')
plt.ylabel('Valori predetti')
plt.show()

# SVM - versione migliore è la 2
svm_v3 = AdaBoostClassifier(random_state=3, base_estimator=svm_definitivo_v2, algorithm='SAMME')
iperparametri = {'n_estimators': list(range(2, 3))}
clf = GridSearchCV(estimator=svm_v3, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v2_scaled, y_train_v2)
print('The best value for parameter n_estimators is', clf.best_params_.get('n_estimators'),
      'since these lead to F1-score =', clf.best_score_)
svm_definitivo_v3 = AdaBoostClassifier(random_state=3, base_estimator=svm_definitivo_v2, n_estimators=10)  # F1-score = 0.4745064536372666
svm_definitivo_v3.fit(X_train_v2_scaled, y_train_v2)
y_pred = svm_definitivo_v3.predict(X_test_v2_scaled)
execute_scores("SVM v3", y_test_v2, y_pred)

mat = confusion_matrix(y_test_v2, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Valori reali')
plt.ylabel('Valori predetti')
plt.show()

# NEURAL NETWORK - la versione migliore è la 2
nn_v3 = AdaBoostClassifier(random_state=3, base_estimator=nn_definitivo_v2)
iperparametri = {'n_estimators': list(range(1, 20))}
clf = GridSearchCV(estimator=nn_v3, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v2_scaled, y_train_v2)
print('The best value for parameter n_estimators is', clf.best_params_.get('n_estimators'),
      'since these lead to F1-score =', clf.best_score_)
nn_definitivo_v3 = AdaBoostClassifier(random_state=3, base_estimator=nn_definitivo_v1, n_estimators=10)  # F1-score = 0.985922839366024
nn_definitivo_v3.fit(X_train_v2_scaled, y_train_v2)
y_pred = nn_definitivo_v3.predict(X_test_v2_scaled)
execute_scores("NN v2", y_test_v2, y_pred)

mat = confusion_matrix(y_test_v2, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('Valori reali')
plt.ylabel('Valori predetti')
plt.show()

########################################################################################################################
# CROSS-VALIDATION PER MODEL ASSESSMENT VERSIONE 3 #
########################################################################################################################
# DECISION TREE
scores = cross_validate(decision_tree_definitivo_v3, X_train_v1, y_train_v1, cv=5,
                        scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of Decision Tree v3 is', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of Decision Tree v3 is ', np.mean(scores['test_balanced_accuracy']))

# SVM
scores = cross_validate(svm_definitivo_v3, X_train_v2_scaled, y_train_v2, cv=5, scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of SVM v3 is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of SVM v3 is ', np.mean(scores['test_balanced_accuracy']))

# NEURAL NETWORK
scores = cross_validate(nn_definitivo_v3, X_train_v1, y_train_v1, cv=5, scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of NN v3 is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of NN v3 is ', np.mean(scores['test_balanced_accuracy']))
