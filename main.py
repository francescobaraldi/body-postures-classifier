import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import classification_report, \
    balanced_accuracy_score, \
    confusion_matrix, \
    precision_score, \
    recall_score, \
    f1_score

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 20)

df = pd.read_csv('datasets/dataset-har-PUC-Rio-ugulino.csv', sep=';')


def kdeplot(feature, labels=list(df['class'].unique())):
    plt.figure(figsize=(10, 6))
    plt.title("KDE for {}".format(feature))
    for label in labels:
        ax = sns.kdeplot(df[df['class'] == label][feature].dropna(), label='Position: {}'.format(label))
        ax.legend()


def barplot_percentages(feature, orient='v', axis_name="percentuale per ogni classe"):
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
    print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, average='weighted'))
    print('Recall:', recall_score(y_test, y_pred, average='weighted'))
    print('F1-Score:', f1_score(y_test, y_pred, average='weighted'))
    # print('AUC is ', roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr'))


def model_assessment(model_name, model, X, y):
    scores = cross_validate(model, X, y, cv=5, scoring=('f1_weighted', 'balanced_accuracy'))
    print('The cross-validated F1-score of ' + model_name + 'is', np.mean(scores['test_f1_weighted']))
    print('The cross-validated Balanced Accuracy of ' + model_name + 'is ', np.mean(scores['test_balanced_accuracy']))


def features_importance(model_name, model, X):
    importance = pd.Series(data=model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 10))
    plt.title("Feature importance for " + model_name)
    sns.barplot(y=importance.index, x=importance.values, palette="Blues_d", orient='h')


def plot_confusion_matrix(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred, normalize=True)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('Valori reali')
    plt.ylabel('Valori predetti')
    plt.show()


# EDA
print(df.head())
print(df.dtypes)
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
df = df.drop(df[df['z4'].isna()].index)  # Elimino la riga con z4 NaN, un'alternativa è correggere il valore
df['z4'] = df['z4'].astype('int64')
df.index = range(0, len(df))  # Ordine corretto degli index

sns.catplot(y="class", kind="count", data=df, height=2.5, aspect=2.5, orient='h')  # Dataset sbilanciato
sns.catplot(y="gender", kind="count", data=df, height=2.5, aspect=2.5, orient='h')
sns.catplot(y="user", kind="count", data=df, height=2.5, aspect=2.5, orient='h')

categorical = ['user', 'gender', 'age', 'how_tall_in_meters', 'weight', 'body_mass_index']
for feat in categorical:
    print(pd.crosstab(df['class'], df[feat], normalize=True))

barplot_percentages("user")

g = sns.FacetGrid(df, row='user', col="class", hue="class", height=3.5)
g.map(plt.scatter, "x1", "x2", alpha=0.6)
g.add_legend()

positional = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4']
for feat in ["x2", "y2", "z2"]:
    kdeplot(feat)

for feat in positional:  # Vedo la correlazione tra classe e valori delle coordinate e i vari outlier
    plt.figure()
    plt.title("Scatterplot for {}".format(feat))
    plt.xlabel("Class")
    plt.ylabel(feat)
    plt.scatter(df['class'], df[feat])

sns.pairplot(df[positional])
g = sns.PairGrid(df, y_vars=positional, x_vars=positional, height=2, hue="class", aspect=1.1)
ax = g.map(plt.scatter, alpha=0.6)
g.add_legend()

X = df.drop(['user', 'gender', 'class'], axis=1)
y = df['class'].values
X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=3)

########################################################################################################################
# CROSS-VALIDATION PER MODEL SELECTION VERSIONE 1#
########################################################################################################################
# DECISION TREE
decision_tree_v1 = tree.DecisionTreeClassifier(random_state=3)
iperparametri = {'max_depth': [5, 10, 20, 50], 'class_weight': ['balanced', None], 'criterion': ['gini', 'entropy'],
                 'splitter': ['best', 'random']}
clf = GridSearchCV(estimator=decision_tree_v1, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v1, y_train_v1)
print('Decision Tree v1')
print('The best value for parameter max_depth is', clf.best_params_.get('max_depth'),
      'the best value for parameter class_weight is', clf.best_params_.get('class_weight'),
      'the best value for parameter criterion is', clf.best_params_.get('criterion'),
      'the best value for parameter splitter is', clf.best_params_.get('splitter'),
      'since it leads to F1-score =', clf.best_score_)  # F1-score = 0.9851278588981345
decision_tree_definitivo_v1 = tree.DecisionTreeClassifier(random_state=3, max_depth=20, class_weight=None,
                                                          criterion='entropy',
                                                          splitter='best')
decision_tree_definitivo_v1.fit(X_train_v1, y_train_v1)
y_pred = decision_tree_definitivo_v1.predict(X_test_v1)
execute_scores("Decision Tree v1", y_test_v1, y_pred)
print(classification_report(y_test_v1, y_pred))

features_importance("Decision Tree v1", decision_tree_definitivo_v1, X_train_v1)
plot_confusion_matrix(y_test_v1, y_pred)

# SVM
svm_v1 = SVC(random_state=3, verbose=True)
iperparametri = {'C': [0.1, 1000], 'kernel': ['linear', 'poly', 'rbf'], 'class_weight': ['balanced', None],
                 'max_iter': [10, 50], 'decision_function_shape': ['ovo', 'ovr']}
clf = GridSearchCV(estimator=svm_v1, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v1, y_train_v1)
print('SVM v1')
print('The best value for parameter C is', clf.best_params_.get('C'),
      'the best value for parameter kernel is', clf.best_params_.get('kernel'),
      'the best value for parameter class_weight is', clf.best_params_.get('class_weight'),
      'the best value for parameter max_iter is', clf.best_params_.get('max_iter'),
      'the best value for parameter decision_function_shape is', clf.best_params_.get('decision_function_shape'),
      'since these lead to F1-score =', clf.best_score_)
svm_definitivo_v1 = SVC(random_state=3, C=1000, kernel='rbf', class_weight='balanced', max_iter=10,
                        decision_function_shape='ovo')  # F1-score = 0.46786182695409206
svm_definitivo_v1.fit(X_train_v1, y_train_v1)
y_pred = svm_definitivo_v1.predict(X_test_v1)
execute_scores("SVM v1", y_test_v1, y_pred)

plot_confusion_matrix(y_test_v1, y_pred)

# NEURAL NETWORK
nn_v1 = MLPClassifier(random_state=3, verbose=True, tol=1e-6)
iperparametri = {'hidden_layer_sizes': [(50, 50), (5, 5, 5, 5), (20, 20, 20)], 'max_iter': [10, 50],
                 'early_stopping': [True, False], 'activation': ['logistic', 'tanh']}
clf = GridSearchCV(estimator=nn_v1, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v1, y_train_v1)
print('NN v1')
print('The best value for parameter hidden_layer_sizes is', clf.best_params_.get('hidden_layer_sizes'),
      'the best value for parameter max_iter is', clf.best_params_.get('max_iter'),
      'the best value for parameter early_stopping is', clf.best_params_.get('early_stopping'),
      'the best value for parameter activation is', clf.best_params_.get('activation'),
      'since these lead to F1-score =', clf.best_score_)
nn_definitivo_v1 = MLPClassifier(random_state=3, tol=1e-6, hidden_layer_sizes=(50, 50), max_iter=50,
                                 early_stopping=False, activation='logistic')  # F1-score = 0.9803673186309148
nn_definitivo_v1.fit(X_train_v1, y_train_v1)
y_pred = nn_definitivo_v1.predict(X_test_v1)
execute_scores("NN v1", y_test_v1, y_pred)

plot_confusion_matrix(y_test_v1, y_pred)

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

# PREPROCESSING
encoder_class = LabelEncoder().fit(df['class'])
y = df['class'] = encoder_class.transform(df['class'])
encoder_gender = LabelEncoder().fit(df['gender'])
df['gender'] = encoder_gender.transform(df['gender'])

# Matrice di correlazione
corr_matrix = df.corr()
sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns, linewidths=.2, cmap="YlGnBu")

features = ['how_tall_in_meters', 'x1', 'y1', 'z1', 'y2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4']
X = df[features]
X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(X, y, test_size=0.2, stratify=y)

for feat in features:
    plt.figure(figsize=(5, 5))
    plt.title("Distplot for {}".format(feat))
    sns.distplot(X_train_v2[feat])
mms = MinMaxScaler().fit(X_train_v2)
X_train_v2_scaled = pd.DataFrame(mms.transform(X_train_v2), columns=features)
X_test_v2_scaled = pd.DataFrame(mms.transform(X_test_v2), columns=features)
for feat in features:
    plt.figure(figsize=(5, 5))
    plt.title("Distplot for {}".format(feat))
    sns.distplot(X_train_v2_scaled[feat])

########################################################################################################################
# CROSS-VALIDATION PER MODEL SELECTION VERSIONE 2 #
########################################################################################################################
# DECISION TREE
decision_tree_v2 = tree.DecisionTreeClassifier(random_state=3)
iperparametri = {'max_depth': [5, 10, 20, 50], 'class_weight': ['balanced', None], 'criterion': ['gini', 'entropy'],
                 'splitter': ['best', 'random']}
clf = GridSearchCV(estimator=decision_tree_v2, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v2_scaled, y_train_v2)
print('Decision Tree v2')
print('The best value for parameter max_depth is', clf.best_params_.get('max_depth'),
      'the best value for class_weight is', clf.best_params_.get('class_weight'),
      'the best value for criterion is', clf.best_params_.get('criterion'),
      'the best value for splitter is', clf.best_params_.get('splitter'),
      'since it leads to balanced F1-score =', clf.best_score_)  # F1-score = 0.9770774608543087
decision_tree_definitivo_v2 = tree.DecisionTreeClassifier(random_state=3, max_depth=50, class_weight=None,
                                                          criterion='entropy',
                                                          splitter='best')  # F1-score = 0.981942065487494
decision_tree_definitivo_v2.fit(X_train_v2_scaled, y_train_v2)
y_pred = decision_tree_definitivo_v2.predict(X_test_v2_scaled)
execute_scores("Decision Tree v2", y_test_v2, y_pred)

features_importance("Decision Tree v2", decision_tree_definitivo_v2, X_train_v2_scaled)
plot_confusion_matrix(y_test_v2, y_pred)

# SVM
pca = PCA(random_state=3, svd_solver='randomized', whiten=True)
svc = SVC(random_state=3, verbose=True)
svm_v2 = make_pipeline(pca, svc)
iperparametri = {'svc__C': [0.1, 1000], 'svc__kernel': ['linear', 'ploy', 'rbf'],
                 'svc__class_weight': ['balanced', None], 'svc__max_iter': [10, 50],
                 'svc__decision_function_shape': ['ovo', 'ovr'], 'pca__n_components': [1, 5, 10, 20]}
clf = GridSearchCV(estimator=svm_v2, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v2_scaled, y_train_v2)
print('SVM v2')
print('The best value for parameter C is', clf.best_params_.get('svc__C'),
      'the best value for parameter kernel is', clf.best_params_.get('svc__kernel'),
      'the best value for parameter class_weight is', clf.best_params_.get('svc__class_weight'),
      'the best value for parameter max_iter is', clf.best_params_.get('svc__max_iter'),
      'the best value for parameter decision_function_shape is', clf.best_params_.get('svc__decision_function_shape'),
      'the best value for parameter n_components is', clf.best_params_.get('pca__n_components'),
      'since these lead to F1-score =', clf.best_score_)  # F1-score = 0.5545493382132174
pca = PCA(random_state=3, svd_solver='randomized', whiten=True, n_components=10)
svc = SVC(random_state=3, C=1000, kernel='rbf', class_weight=None, max_iter=50, decision_function_shape='ovo')
svm_definitivo_v2 = make_pipeline(pca, svc)
svm_definitivo_v2.fit(X_train_v2_scaled, y_train_v2)
y_pred = svm_definitivo_v2.predict(X_test_v2_scaled)
execute_scores("SVM v2", y_test_v2, y_pred)

plot_confusion_matrix(y_test_v2, y_pred)

# NEURAL NETWORK
nn_v2 = MLPClassifier(random_state=3, verbose=True, tol=1e-6)
iperparametri = {'hidden_layer_sizes': [(50, 50), (20, 20, 20), (5, 5, 5, 5)], 'max_iter': [10, 50],
                 'early_stopping': [True, False], 'activation': ['logistic', 'tanh']}
clf = GridSearchCV(estimator=nn_v2, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v2_scaled, y_train_v2)
print('NN v2')
print('The best value for parameter hidden_layer_sizes is', clf.best_params_.get('hidden_layer_sizes'),
      'the best value for parameter max_iter is', clf.best_params_.get('max_iter'),
      'the best value for parameter early_stopping is', clf.best_params_.get('early_stopping'),
      'the best value for parameter activation is', clf.best_params_.get('activation'),
      'since these lead to F1-score =', clf.best_score_)
nn_definitivo_v2 = MLPClassifier(random_state=3, tol=1e-6, hidden_layer_sizes=(20, 20, 20), max_iter=50,
                                 early_stopping=True, activation='tanh')  # F1-score = 0.9612301061404441
nn_definitivo_v2.fit(X_train_v2_scaled, y_train_v2)
y_pred = nn_definitivo_v2.predict(X_test_v2_scaled)
execute_scores("NN v2", y_test_v2, y_pred)

plot_confusion_matrix(y_test_v2, y_pred)

# NEURAL NETWORK con PCA
pca = PCA(random_state=3, svd_solver='randomized', whiten=True)
nn = MLPClassifier(random_state=3, verbose=True, tol=1e-6)
nn_v2 = make_pipeline(pca, nn)
iperparametri = {'mlpclassifier__hidden_layer_sizes': [(50, 50), (20, 20, 20), (5, 5, 5, 5)],
                 'mlpclassifier__max_iter': [10, 50], 'mlpclassifier__early_stopping': [True, False],
                 'mlpclassifier__activation': ['logistic', 'tanh'], 'pca__n_components': [1, 5, 10, 20]}
clf = GridSearchCV(estimator=nn_v2, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v2_scaled, y_train_v2)
print('NN v2')
print('The best value for parameter hidden_layer_sizes is', clf.best_params_.get('mlpclassifier__hidden_layer_sizes'),
      'the best value for parameter max_iter is', clf.best_params_.get('mlpclassifier__max_iter'),
      'the best value for parameter early_stopping is', clf.best_params_.get('mlpclassifier__early_stopping'),
      'the best value for parameter activation is', clf.best_params_.get('mlpclassifier__activation'),
      'the best value for parameter n_components is', clf.best_params_.get('pca__n_components'),
      'since these lead to F1-score =', clf.best_score_)  # F1-score = 0.9909297411999592
pca = PCA(random_state=3, svd_solver='randomized', whiten=True, n_components=10)
nn = MLPClassifier(random_state=3, verbose=True, tol=1e-6, hidden_layer_sizes=(50, 50), max_iter=50,
                   early_stopping=False, activation='tanh')
nn_definitivo_v2 = make_pipeline(pca, nn)
nn_definitivo_v2.fit(X_train_v2_scaled, y_train_v2)
y_pred = nn_definitivo_v2.predict(X_test_v2_scaled)
execute_scores("NN v2", y_test_v2, y_pred)

plot_confusion_matrix(y_test_v2, y_pred)

########################################################################################################################
# CROSS-VALIDATION PER MODEL ASSESSMENT VERSIONE 2 #
########################################################################################################################
# DECISION TREE
scores = cross_validate(decision_tree_definitivo_v2, X_train_v2_scaled, y_train_v2, cv=5,
                        scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of Decision Tree v2 is', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of Decision Tree v2 is ', np.mean(scores['test_balanced_accuracy']))

# SVM
scores = cross_validate(svm_definitivo_v2, X_train_v2_scaled, y_train_v2, cv=5,
                        scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of SVM v2 is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of SVM v2 is ', np.mean(scores['test_balanced_accuracy']))

# NEURAL NETWORK
scores = cross_validate(nn_definitivo_v2, X_train_v2_scaled, y_train_v2, cv=5,
                        scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of NN v2 is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of NN v2 is ', np.mean(scores['test_balanced_accuracy']))

########################################################################################################################
# CROSS-VALIDATION PER MODEL SELECTION VERSIONE 3 #
########################################################################################################################
# Decision Tree - versione migliore è la 1
decision_tree_v3 = RandomForestClassifier(random_state=3, max_depth=20, class_weight=None, criterion='entropy')
iperparametri = {'n_estimators': [5, 10, 20, 50]}
clf = GridSearchCV(estimator=decision_tree_v3, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v1, y_train_v1)
print('Decision Tree v3')
print('The best value for parameter n_estimators is', clf.best_params_.get('n_estimators'),
      'since it leads to balanced F1-score =', clf.best_score_)
decision_tree_definitivo_v3 = RandomForestClassifier(random_state=3, max_depth=20, class_weight=None,
                                                     criterion='entropy',
                                                     n_estimators=50)  # F1-score = 0.9958063054386616
decision_tree_definitivo_v3.fit(X_train_v1, y_train_v1)
y_pred = decision_tree_definitivo_v3.predict(X_test_v1)
execute_scores("Decision Tree v3", y_test_v1, y_pred)

features_importance("Decision Tree v3", decision_tree_definitivo_v3, X_train_v1)
plot_confusion_matrix(y_test_v1, y_pred)

# SVM - versione migliore è la 2
svm_v3 = BaggingClassifier(random_state=3, base_estimator=svm_definitivo_v2, verbose=1)
iperparametri = {'n_estimators': [10, 50, 100]}
clf = GridSearchCV(estimator=svm_v3, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v2_scaled, y_train_v2)
print('SVM v3')
print('The best value for parameter n_estimators is', clf.best_params_.get('n_estimators'),
      'since these lead to F1-score =', clf.best_score_)
svm_definitivo_v3 = BaggingClassifier(random_state=3, base_estimator=svm_definitivo_v2, verbose=1,
                                      n_estimators=100)  # F1-score = 0.6147914579142865
svm_definitivo_v3.fit(X_train_v2_scaled, y_train_v2)
y_pred = svm_definitivo_v3.predict(X_test_v2_scaled)
execute_scores("SVM v3", y_test_v2, y_pred)

plot_confusion_matrix(y_test_v2, y_pred)

# NEURAL NETWORK - la versione migliore è la 1
nn_v3 = BaggingClassifier(random_state=3, base_estimator=nn_definitivo_v1, verbose=1)
iperparametri = {'n_estimators': [10, 20, 50]}
clf = GridSearchCV(estimator=nn_v3, param_grid=iperparametri, scoring='f1_weighted', cv=5)
clf.fit(X_train_v1, y_train_v1)
print('NN v3')
print('The best value for parameter n_estimators is', clf.best_params_.get('n_estimators'),
      'since these lead to F1-score =', clf.best_score_)  # F1-score = 0.9898294941897781
nn_definitivo_v3 = BaggingClassifier(random_state=3, base_estimator=nn_definitivo_v2, verbose=1,
                                     n_estimators=20)
nn_definitivo_v3.fit(X_train_v2_scaled, y_train_v2)
y_pred = nn_definitivo_v3.predict(X_test_v2_scaled)
execute_scores("NN v3", y_test_v2, y_pred)

mat = confusion_matrix(y_test_v1, y_pred, normalize=True)
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
scores = cross_validate(svm_definitivo_v3, X_train_v2_scaled, y_train_v2, cv=5,
                        scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of SVM v3 is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of SVM v3 is ', np.mean(scores['test_balanced_accuracy']))

# NEURAL NETWORK
scores = cross_validate(nn_definitivo_v3, X_train_v1, y_train_v1, cv=5, scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of NN v3 is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of NN v3 is ', np.mean(scores['test_balanced_accuracy']))
