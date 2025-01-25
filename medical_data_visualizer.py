import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add overweight column by calculating BMI and comparing to standard
df['overweight'] = np.where(df['weight']/((df['height']/100)**2) > 25, 1, 0)

# Normalized data to binary system where 0 is always the "good" value (normal cholesterol/glucose levels)
# and 1 is always the "bad" value (abnormal cholesterol/glucose values)
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)

# Draw and save a categorical plot based on normalized data
def draw_cat_plot():
    # Melt data to new dataframe using only categorical values indexed by presence of cardiovascular disease
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=df.loc[:, 'cholesterol':'overweight'].columns)

    # Group & reformat data to split by 0 and 1 values for cardio
    # Show counts of each feature for both subsets of data
    # Convert to long format
    df_cat = pd.DataFrame(df_cat.groupby(by=['cardio', 'variable', 'value']).value_counts(), columns=['total'])

    # Draw categorical plots for each of the two subsets of data
    fig, ax = plt.subplots()
    fig = sns.catplot(data=df_cat, x='variable', y='total', hue='value', kind='bar', col='cardio')

    # Save figure as .png file and return it from the function for testing
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = None

    # 12
    corr = None

    # 13
    mask = None



    # 14
    fig, ax = None

    # 15



    # 16
    fig.savefig('heatmap.png')
    return fig
