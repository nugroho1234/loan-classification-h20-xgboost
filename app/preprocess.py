#basic libraries
import numpy as np
import pandas as pd
import re
import os
import joblib

#h2o
import h2o

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
    if len(extracted_numbers) > 0:
        df[column] = extracted_numbers
    column_dtype = df[column].dtype
    if column_dtype == 'object' or column_dtype == 'float32':
        df[column] = df[column].astype('float64')
    return df

def has_non_ascii(s):
    for char in s:
        if ord(char) > 127:
            return True
    return False

def create_dataframe(data, knn_initial_model, purpose_mapping, knn_cur_model, scaler):
    df = pd.DataFrame([data])
        
    #clean credit score
    df.loc[df['credit_score'] > 850, 'credit_score'] = df.loc[df['credit_score'] > 850, 'credit_score'] / 10
    
    #clean home ownership
    df['home_ownership'] = df['home_ownership'].replace('HaveMortgage', 'Home Mortgage')
    
    #convert string values into lower case and snake case
    df = df.applymap(lambda x: x if not isinstance(x, str) or not has_non_ascii(x) else x.encode('ascii', 'ignore').decode('ascii'))
    categorical_cols = ['term', 'home_ownership', 'purpose']
    for col in categorical_cols:
        df[col] = clean_columns(df[col].tolist())
    
    #convert term 
    term_dict = {'short_term':0, 'long_term':1}
    df.replace({"term": term_dict}, inplace=True)
    
    #impute missing values if any
    column_names_to_impute = ['current_loan_amount', 'credit_score', 'years_in_current_job', 'annual_income', 'months_since_last_delinquent', 'maximum_open_credit', 'bankruptcies', 'tax_liens']
    column_with_missing_values = df.columns[df.isnull().any()].tolist()
    imputed = knn_initial_model.transform(df[column_names_to_impute].values)
    data_temp = pd.DataFrame(imputed, columns=column_names_to_impute, index = df.index)
    df[column_with_missing_values] = data_temp[column_with_missing_values]
    
    #simplify purpose
    df['purpose'] = df['purpose'].map(purpose_mapping)
    df['purpose'].fillna('other', inplace=True)
    
    #feature engineering
    df['debt_equity_ratio'] = df['monthly_debt'] / df['annual_income']
    df['credit_utilization_ratio'] = df['current_credit_balance'] / df['maximum_open_credit']
    df['is_months_delinquent_missing'] = df['months_since_last_delinquent'].isnull().astype(int)
    df['has_stable_job'] = (df['years_in_current_job'] > 2).astype(int)
    
    #drop unneeded columns
    df.drop(['bankruptcies', 'monthly_debt', 'annual_income', 'current_credit_balance', 'years_in_current_job', 'maximum_open_credit', 'months_since_last_delinquent', 'tax_liens'], axis = 1, inplace = True)
    
    #impute credit_utilization_ratio if needed
    column_with_missing_values = df.columns[df.isnull().any()].tolist()
    if len(column_with_missing_values) > 0:
        column_names_to_impute = ['credit_utilization_ratio']
        df = df.replace([np.inf, -np.inf], np.nan)
        imputed = knn_cur_model.transform(df[column_names_to_impute].values)
        data_temp = pd.DataFrame(imputed, columns=column_names_to_impute, index = df.index)
        df[column_names_to_impute] = data_temp
    else:
        pass
    
    #one hot encode purpose and home_ownership
    #dummy variable names for purpose and home_ownership in the expected dataframe 
    all_purpose_cols = ['purpose_debt_consolidation', 'purpose_other', 'purpose_personal_loans'] 
    all_home_cols = ['home_own_home', 'home_rent']
    
    #one hot encode
    new_dummies_purpose = pd.get_dummies(df['purpose'], prefix='purpose').replace({True: 1, False: 0})
    new_dummies_home = pd.get_dummies(df['home_ownership'], prefix='home').replace({True: 1, False: 0})
    list_dummies_purpose = list(new_dummies_purpose.columns)
    list_dummies_home = list(new_dummies_home.columns)
    
    #create similar column to expected dataframe
    for col in all_purpose_cols:
        if col not in new_dummies_purpose.columns:
            new_dummies_purpose[col] = 0
    for col in all_home_cols:
        if col not in new_dummies_home.columns:
            new_dummies_home[col] = 0
    
    #drop first dummies if neccessary
    for col in list_dummies_purpose:
        if col in all_purpose_cols:
            pass
        else:
            new_dummies_purpose.drop(col, axis=1,inplace=True)

    for col in list_dummies_home:
        if col in list(all_home_cols):
            pass
        else:
            new_dummies_home.drop(col, axis=1,inplace=True)
    
    #change the values into the dummy variables
    df[new_dummies_purpose.columns] = new_dummies_purpose
    df[new_dummies_home.columns] = new_dummies_home
    
    #drop home_ownership and purpose
    df.drop(["home_ownership", "purpose"], axis=1, inplace = True)
    
    #standardize numeric values except for binaries and ratios
    cols_to_standardize = ['current_loan_amount', 'credit_score', 'years_of_credit_history', 'number_of_open_accounts', 'number_of_credit_problems']
    data_scaled = scaler.transform(df[cols_to_standardize].values)
    data_temp = pd.DataFrame(data_scaled, columns=cols_to_standardize, index = df.index)
    df[cols_to_standardize] = data_temp
    
    return df

def predict_data(model, hf, threshold_metrics='precision'):
    #predict
    predictions = model.predict(hf)
    
    #convert to pandas dataframe
    prediction_df = predictions.as_data_frame()
    
    #applying threshold to the prediction
    loan_given_prob = prediction_df['loan_given'].tolist()[0]
    #loan_refused_prob = prediction_df['loan_refused'].tolist()[0]
    threshold = model.find_threshold_by_max_metric(threshold_metrics)
    
    loan_prediction = [('loan_given' if loan_given_prob > threshold else 'loan_refused').replace('_', ' ').title()]
    
    #create prediction dictionary for json
    prediction_dict = {}
    prediction_dict['prediction'] = loan_prediction
    return prediction_dict