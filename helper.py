#basic libraries
import numpy as np
import pandas as pd
import re

#scikit-learn libraries
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

#h2o models and grid search
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch

#plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

def create_bar_chart(df,col,rotation=0):
    value_counts_series = df[col].value_counts()

    # Create a bar chart using Seaborn
    sns.barplot(x=value_counts_series.index, y=value_counts_series.values)

    # Add labels and title
    plt.xlabel(col.replace('_',' ').title())
    plt.ylabel('Count')
    plt.title('Distribution of ' + col.replace('_',' ').title())

    plt.xticks(rotation=rotation)

    # Show the plot
    plt.show();

def create_histogram(df,col):
    plt.hist(df[col])
    plt.title('Histogram of ' + col.replace('_',' ').title())
    plt.show()
    
def clean_columns(column_list):
    all_cols = column_list
    
    modified_list = []

    for item in all_cols:
        item = str(item).lower()
        modified_item = re.sub(r'[^a-zA-Z0-9]', '_', item)
        modified_list.append(modified_item)
    
    final_list = []
    
    for i in modified_list:
        cleaned_column_name = re.sub(r'_+', '_', i)
        final_list.append(cleaned_column_name)
    
    final_list = [col.strip('_') for col in final_list]
        
    return final_list

def create_stacked_barcharts(df, x, y):
    # Create non-normalized crosstab
    table = pd.crosstab(df[x], df[y])
    
    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot non-normalized stacked bar chart
    colors = plt.cm.tab20.colors
    bottom = [0] * len(table.index)
    
    for idx, category in enumerate(table.columns):
        axes[0].bar(
            table.index,
            table[category],
            bottom=bottom,
            label=category,
            color=colors[idx],
        )
        bottom += table[category]
    
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Stacked Barchart of {x} vs {y} (Non-normalized)')
    axes[0].legend()
    
    # Create normalized crosstab
    normalized_table = pd.crosstab(df[x], df[y], normalize='index')
    
    # Plot normalized stacked bar chart
    bottom = [0] * len(normalized_table.index)
    
    for idx, category in enumerate(normalized_table.columns):
        axes[1].bar(
            normalized_table.index,
            normalized_table[category],
            bottom=bottom,
            label=category,
            color=colors[idx],
        )
        bottom += normalized_table[category]
    
    axes[1].set_ylabel('Percentage')
    axes[1].set_title(f'Stacked Barchart of {x} vs {y} (Normalized)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return table, normalized_table

def create_facet_grid(df, continuous_vars, target):
    # Standardize the continuous variables
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[continuous_vars])

    # Create a new DataFrame with the standardized values
    df_standardized = pd.DataFrame(df_scaled, columns=continuous_vars)

    # Add target column back to the DataFrame
    df_standardized[target] = df[target]

    # Melt the data for FacetGrid
    df_melted = df_standardized.melt(id_vars=[target], value_vars=continuous_vars, var_name='variable', value_name='value')

    # Create a FacetGrid with violin plots
    g = sns.FacetGrid(df_melted, col=target, col_wrap=2, height=6)
    g.map(sns.violinplot, 'variable', 'value', inner='quartile', palette='Set1')
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("", "Standardized Value")
    g.set_xticklabels(rotation=45)
    g.tight_layout()
    plt.suptitle(f"Violin Plots of Standardized Continuous Variables by {target.replace('_', ' ').title()}", y=1.02)
    plt.show();

def clip_outliers(df, col: str):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    upper_threshold = q3 + (1.5 * (q3 - q1))
    df[col] = df[col].clip(upper=upper_threshold)

def create_box_violin_plot(df,x,y):
    fig,axes=plt.subplots(1,2,figsize=(18,10))
    fig.suptitle("Violin and box plots for variable : {}".format(y))
    sns.violinplot(ax=axes[0],x=x,y=y,data=df)
    sns.boxplot(ax=axes[1],data=df[y])

    axes[0].set_title(f"Violin plot for variable {y} and its relation with {x}")
    axes[1].set_title(f"Box plot for variable {y}")
    
    plt.show();

