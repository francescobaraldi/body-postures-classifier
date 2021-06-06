import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 20)

df = pd.read_csv('datasets/dataset-har-PUC-Rio-ugulino.csv', sep=';')

# EDA
print(df.head())
print(df.shape)
print(df.columns)
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())

df['body_mass_index'] = df['body_mass_index'].str.replace(',', '.')
df['body_mass_index'] = df['body_mass_index'].astype('float64')
df['how_tall_in_meters'] = df['how_tall_in_meters'].str.replace(',', '.')
df['how_tall_in_meters'] = df['how_tall_in_meters'].astype('float64')
df['z4'] = pd.to_numeric(df['z4'], errors='coerce')
df = df.drop(df[df['z4'].isna()].index)  # Elimino la riga con z4 NaN, alternativa è cambiare il valore
df['z4'] = df['z4'].astype('int64')

ax = sns.catplot(y="class", kind="count", data=df, height=2.6, aspect=2.5, orient='h')  # dataset sbilanciato
ax = sns.catplot(y="age", kind="count", data=df, height=2.6, aspect=2.5, orient='h')
ax = sns.catplot(y="gender", kind="count", data=df, height=2.6, aspect=2.5, orient='h')


def kdeplot(feature, labels=list(df['class'].unique())):
    plt.figure(figsize=(10, 6))
    plt.title("KDE for {}".format(feature))
    for label in labels:
        ax = sns.kdeplot(df[df['class'] == label][feature].dropna(), label='Churn: {}'.format(label))


kdeplot("x2", ["standing"])


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


barplot_percentages("age")
barplot_percentages("gender")
barplot_percentages("how_tall_in_meters")
barplot_percentages("weight")
barplot_percentages("body_mass_index")

# Matrice di correlazione
corr_matrix = df.corr()
sns.heatmap(corr_matrix)
