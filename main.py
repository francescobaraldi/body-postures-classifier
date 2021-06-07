import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 20)

df = pd.read_csv('datasets/dataset-har-PUC-Rio-ugulino.csv', sep=';')

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

sns.catplot(y="class", kind="count", data=df, height=2.6, aspect=2.5, orient='h')  # dataset sbilanciato
sns.catplot(y="gender", kind="count", data=df, height=2.6, aspect=2.5, orient='h')
sns.catplot(y="user", kind="count", data=df, height=2.6, aspect=2.5, orient='h')


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

sns.barplot(x="age", y=df['user'], hue='user', data=df, orient='v')

feat = list(set(df.columns) - {'user', 'class'})
sns.pairplot(df[feat], hue='class')

df['gender'], _ = pd.factorize(df['gender'])

# Matrice di correlazione
corr_matrix = df.corr()
sns.heatmap(corr_matrix)

features = ['gender', 'how_tall_in_meters', 'weight', 'body_mass_index', 'x2', 'y2', 'z2']
X = df[features].values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CROSS-VALIDATION PER MODEL SELECTION
knn_model = KNeighborsClassifier()
hyperparameters = {'n_neighbors': list(range(1, 10))}
model = knn_model
clf = GridSearchCV(estimator=model, param_grid=hyperparameters, scoring='f1_weighted', cv=5)
clf.fit(X_train, y_train)
print('K-Nearest Neighbor')
print('Overall, the best value for parameter k is ', clf.best_params_.get('n_neighbors'),
      ' since it leads to F1-score = ', clf.best_score_)

nn_model = MLPClassifier(verbose=True)
hyperparameters = {'hidden_layer_sizes': [(100,), (100, 100)], 'max_iter': [200], 'activation': ['logistic']}
model = nn_model
clf = GridSearchCV(estimator=model, param_grid=hyperparameters, scoring='balanced_accuracy', cv=5)
clf.fit(X_train, y_train)
print('Overall, the best choice for parameter hidden_layer_sizes is ', clf.best_params_.get('hidden_layer_sizes'),
      'the best choice for parameter max_iter is ', clf.best_params_.get('max_iter'),
      'the best value for parameter activation is ', clf.best_params_.get('activation'),
      'since these lead to accuracy = ', clf.best_score_)
