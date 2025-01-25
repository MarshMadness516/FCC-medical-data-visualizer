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
    cat_plot = sns.catplot(data=df_cat, x='variable', y='total', hue='value', kind='bar', col='cardio')

    # Save figure as .png file and return it from the function for testing
    fig = cat_plot.fig
    fig.savefig('catplot.png')
    return fig


# Draw and save a heat map based on normalized data
def draw_heat_map():
    # Clean the data
    # Filter out incorrect data in ap_hi, ap_lo, height, and weight
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate correlation matrix for cleaned data
    corr = df_heat.corr()

    # Generate mask for the upper triangle of the correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12,9))

    # Plot and style the heatmap with seaborn using the correlation matrix and mask
    sns.heatmap(
        ax = ax,
        data = corr,
        mask = mask,
        vmax = .32,
        vmin = -.16,
        robust = True,
        annot = True,
        fmt = '.1f',
        linewidth = 0.5,
        center = 0,
        cbar_kws = {
            'shrink': 0.5,
            'ticks': [-0.08, 0.0, 0.08, 0.16, 0.24]
            }
        )

    # Save figure as .png file and return it from the function for testing
    fig = ax.get_figure()
    fig.savefig('heatmap.png')
    return fig
