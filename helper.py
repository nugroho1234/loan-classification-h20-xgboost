#basic libraries
import numpy as np
import pandas as pd
import re

#scikit-learn libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

#h2o models and grid search
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch

#xgboost model
from xgboost import XGBClassifier
import shap

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

def has_non_ascii(s):
    for char in s:
        if ord(char) > 127:
            return True
    return False

def create_box_violin_plot(df,x,y):
    fig,axes=plt.subplots(1,2,figsize=(18,10))
    fig.suptitle("Violin and box plots for variable : {}".format(y))
    sns.violinplot(ax=axes[0],x=x,y=y,data=df)
    sns.boxplot(ax=axes[1],data=df[y])

    axes[0].set_title(f"Violin plot for variable {y} and its relation with {x}")
    axes[1].set_title(f"Box plot for variable {y}")
    
    plt.show();

def calculate_metrics_summary(h2o_perf, df, model_name):
    #calculate precision, recall, and support for class 0 and 1
    precision_class_0 = h2o_perf.precision()[0][0]
    precision_class_1 = h2o_perf.precision()[0][1]
    recall_class_0 = h2o_perf.recall()[0][0]
    recall_class_1 = h2o_perf.recall()[0][1]
    support_class_0 = df.loan_status.value_counts()['loan_refused']
    support_class_1 = df.loan_status.value_counts()['loan_given']
    
    #calculate weighted precision and recall
    precision = (precision_class_0 * support_class_0 + precision_class_1 * support_class_1) / (support_class_0 + support_class_1)
    recall = (recall_class_0 * support_class_0 + recall_class_1 * support_class_1) / (support_class_0 + support_class_1)
    
    #calculate F1 score for class 0 and class 1
    f1_class_0 = 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0)
    f1_class_1 = 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1)
    
    f1 = (f1_class_0 * support_class_0 + f1_class_1 * support_class_1) / (support_class_0 + support_class_1)
    auc = h2o_perf.auc()
    
    print(f"The F1 score for {model_name} is {f1}")
    print(f"The precision score for {model_name} is {precision}")
    print(f"The recall score for {model_name} is {recall}")
    print(f"The AUC score for {model_name} is {auc}")
    
    return f1, precision, recall, auc

def compare_score(f1_scores, precision_scores, recall_scores, auc_scores, models):
    # Bar chart for F1 Score
    plt.figure(figsize=(12, 6))
    plt.subplot(221)
    plt.bar(models, f1_scores, color='skyblue')
    plt.title('F1 Score')
    plt.ylim(0, 1)  # Set appropriate limits

    # Bar chart for Precision
    plt.subplot(222)
    plt.bar(models, precision_scores, color='lightgreen')
    plt.title('Precision')
    plt.ylim(0, 1)  # Set appropriate limits

    # Bar chart for Recall
    plt.subplot(223)
    plt.bar(models, recall_scores, color='lightcoral')
    plt.title('Recall')
    plt.ylim(0, 1)  # Set appropriate limits

    # Bar chart for AUC Score
    plt.subplot(224)
    plt.bar(models, auc_scores, color='gold')
    plt.title('AUC Score')
    plt.ylim(0, 1)  # Set appropriate limits

    plt.tight_layout()
    plt.show();
    
    data = {
        'Model': models,
        'F1 Score': f1_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'AUC Score': auc_scores
    }

    # Create a DataFrame
    scores_df = pd.DataFrame(data)
    return scores_df

def train_gbm_h2o(predictors, response, train, test, val, learn_rate, max_depth, ntrees, sample_rate, col_sample_rate):
    #hyperparameters to tune
    gbm_params1 = {'learn_rate': learn_rate,
                'max_depth': max_depth,
                'ntrees': ntrees,
                'sample_rate': sample_rate,
                'col_sample_rate': col_sample_rate}

    #initializing grid search
    gbm_grid1 = H2OGridSearch(
        model=H2OGradientBoostingEstimator,
        grid_id='gbm_grid1',
        hyper_params=gbm_params1
    )
    
    #train the model
    gbm_grid1.train(
        x=predictors,
        y=response,
        training_frame=train,
        validation_frame=val,  # Include your validation frame here
        seed=42
    )
    
    gbm_gridperf1 = gbm_grid1.get_grid(sort_by='aucpr', decreasing=True)
    
    #get the best model
    best_gbm1 = gbm_gridperf1.models[0]

    #evaluate the model against the test data
    gbm_perf = best_gbm1.model_performance(test)
    
    return best_gbm1, gbm_perf, gbm_gridperf1

def train_xgboost(df, n_estimators, max_depth, subsample, reg_alpha, reg_lambda, learning_rate, max_bin):
    #separate predictors and target
    X = df.drop('loan_status', axis=1)
    y = pd.DataFrame(df['loan_status'])

    #mapping the target variable to 0 and 1 so that the algorithm works
    mapping = {
        'loan_refused': 0,
        'loan_given': 1
    }
    y['loan_status'] = y['loan_status'].map(mapping)

    #split the dataset into train, test, and validation
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, random_state=42)

    print(f'The number of training dataset is {X_train.shape[0]}, the number of validation dataset is {X_val.shape[0]}, and the number of testing dataset is {X_test.shape[0]}.')
    
    param_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'subsample': subsample,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'learning_rate': learning_rate,
        'max_bin': max_bin
    }

    # Initialize the XGBoost classifier
    xgb = XGBClassifier(random_state=42, eval_metric='auc', early_stopping_rounds=10)

    # Initialize GridSearchCV with the XGBoost classifier and parameter grid
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=1)

    # Fit the grid search to the training data
    print('Training the model...')
    grid_search.fit(X_train, y_train,eval_set=[(X_val, y_val)], verbose = 0)
    print('DONE!')
    print("Best Parameters:", grid_search.best_params_)
    best_xgb_model = grid_search.best_estimator_
    
    return best_xgb_model, X_test, y_test

def calculate_model_metrics_xgboost(xgboost_model, X_test, y_test, model_name):
    y_pred = xgboost_model.predict(X_test)
    
    confusion = confusion_matrix(y_test, y_pred)

    class_names = ['loan_refused', 'loan_given']

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Actual vs Predicted)')
    plt.show();

    # Calculate precision, recall, AUC, and F1 score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred)  
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"The F1 score for {model_name} is {f1}")
    print(f"The precision score for {model_name} is {precision}")
    print(f"The recall score for {model_name} is {recall}")
    print(f"The AUC score for {model_name} is {auc}")
    
    return f1, precision, recall, auc

def train_dl_h2o(predictors, response, train, test, val, hidden, epochs, balance_classes, activation):
    #hyperparameters to tune
    hyper_params = {
        'hidden': hidden,
        'epochs': epochs,
        'balance_classes': balance_classes,
        'activation': activation
    }

    #initialize grid search
    dl_grid = H2OGridSearch(
        model=H2ODeepLearningEstimator,
        grid_id='dl_grid',
        hyper_params=hyper_params
    )
    #train the model
    dl_grid.train(
        x=predictors,
        y=response,
        training_frame=train,
        validation_frame=val
    )
    
    # Get the best model from the grid search
    best_dl_model = dl_grid.get_grid()[0]

    #evaluate model performance against the test data
    dl_perf = best_dl_model.model_performance(test)
    
    return best_dl_model, dl_perf

def check_object_column(df, column):
    pattern = r'\$?(\d+\.\d+)'
    column_values = df[column].tolist()
    extracted_numbers = []
    supposed_values = None
    if column in ['current_loan_amount', 'number_of_open_accounts', 'number_of_credit_problems', 'current_credit_balance']:
        supposed_values = 'int'
    elif column in ['credit_score', 'annual_income', 'monthly_debt', 'years_of_credit_history', 'months_since_last_delinquent', 'bankruptcies', 'tax_liens']:
        supposed_values = 'float'
    else:
        supposed_values = 'object'
    column_dtype = df[column].dtype
    if column_dtype == 'object':
        data_list = df[column].tolist()
        for data in data_list:
            match = re.search(pattern, data)
            if match:
                number_str = match.group(1)
                if supposed_values == 'int':
                    extracted_numbers.append(int(number_str))
                elif supposed_values == 'float':
                    extracted_numbers.append(float(number_str))
            else:
                if column == 'monthly_debt':
                    pass
                else:
                    extracted_numbers.append(np.nan)
    else:
        df[column] = column_values
    #print(extracted_numbers)
    if len(extracted_numbers) > 0:
        df[column] = extracted_numbers
    column_dtype = df[column].dtype
    if column_dtype == 'object' or column_dtype == 'float32':
        df[column] = df[column].astype('float64')
    return df

