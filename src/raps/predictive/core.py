import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def execute_glm_regression(elr_dataframe_df, elr_outcome_str, elr_predictors_list,
                           model_type='linear', print_results=True, labels=False, reg_type="Multi"):
    """
    Executes a GLM (Generalized Linear Model) for linear or logistic regression.

    Parameters:
    - elr_dataframe_df (pd.DataFrame): DataFrame containing the data.
    - elr_outcome_str (str): Name of the outcome variable.
    - elr_predictors_list (list): List of predictor variable names.
    - model_type (str): 'linear' for linear regression or 'logistic' for logistic regression.
    - print_results (bool): If True, prints the formatted results table.
    - labels (dict or bool): Optional dictionary to map variable names to readable labels.
    - reg_type (str): 'uni' or 'multi' to control output column naming.

    Returns:
    - summary_df (pd.DataFrame): Formatted summary table with regression results.
    """

    # 1. Define the family based on model_type
    if model_type.lower() == 'logistic':
        family = sm.families.Binomial()
    elif model_type.lower() == 'linear':
        family = sm.families.Gaussian()
    else:
        raise ValueError("model_type must be 'linear' or 'logistic'")

    # 2. Build the formula string
    formula = elr_outcome_str + ' ~ ' + ' + '.join(elr_predictors_list)

    # 3. Convert categorical variables
    categorical_vars = elr_dataframe_df.select_dtypes(include=['object', 'category']).columns.intersection(elr_predictors_list)
    for var in categorical_vars:
        elr_dataframe_df[var] = elr_dataframe_df[var].astype('category')

    # 4. Fit the GLM model
    model = smf.glm(formula=formula, data=elr_dataframe_df, family=family)
    result = model.fit()

    # 5. Extract summary table
    summary_table = result.summary2().tables[1].copy()

    # 6. Format results
    if model_type.lower() == 'logistic':
        summary_table['Odds Ratio'] = np.exp(summary_table['Coef.'])
        summary_table['IC Low'] = np.exp(summary_table['[0.025'])
        summary_table['IC High'] = np.exp(summary_table['0.975]'])

        summary_df = summary_table[['Odds Ratio', 'IC Low', 'IC High', 'P>|z|']].reset_index()
        summary_df = summary_df.rename(columns={'index': 'Study',
                                                'Odds Ratio': 'OddsRatio',
                                                'IC Low': 'LowerCI',
                                                'IC High': 'UpperCI',
                                                'P>|z|': 'p-value'})
    else:
        summary_df = summary_table[['Coef.', '[0.025', '0.975]', 'P>|z|']].reset_index()
        summary_df = summary_df.rename(columns={'index': 'Study',
                                                'Coef.': 'Coefficient',
                                                '[0.025': 'LowerCI',
                                                '0.975]': 'UpperCI',
                                                'P>|z|': 'p-value'})

    # 7. Apply labels if provided
    if labels:
        def parse_variable_name(var_name):
            if var_name == 'Intercept':
                return labels.get('Intercept', 'Intercept')
            elif '[' in var_name:
                base_var = var_name.split('[')[0]
                level = var_name.split('[')[1].split(']')[0]
                base_var_name = base_var.replace('C(', '').replace(')', '').strip()
                label = labels.get(base_var_name, base_var_name)
                return f'{label} ({level})'
            else:
                var_name_clean = var_name.replace('C(', '').replace(')', '').strip()
                return labels.get(var_name_clean, var_name_clean)
        summary_df['Study'] = summary_df['Study'].apply(parse_variable_name)

    # 8. Reorder columns
    if model_type.lower() == 'logistic':
        summary_df = summary_df[['Study', 'OddsRatio', 'LowerCI', 'UpperCI', 'p-value']]
    else:
        summary_df = summary_df[['Study', 'Coefficient', 'LowerCI', 'UpperCI', 'p-value']]

    # 9. Clean categorical variable names
    summary_df['Study'] = summary_df['Study'].str.replace('T.', '')

    # 10. Format numeric values
    for col in summary_df.columns[1:-1]:
        summary_df[col] = summary_df[col].round(3)
    summary_df['p-value'] = summary_df['p-value'].apply(lambda x: f'{x:.4f}')

    # 11. Remove intercept row
    summary_df = summary_df[summary_df['Study'] != 'Intercept']

    # 12. Rename columns based on regression type
    if reg_type.lower() == 'uni':
        if model_type.lower() == 'logistic':
            summary_df.rename(columns={
                'OddsRatio': 'OddsRatio (uni)',
                'LowerCI': 'LowerCI (uni)',
                'UpperCI': 'UpperCI (uni)',
                'p-value': 'p-value (uni)'
            }, inplace=True)
        else:
            summary_df.rename(columns={
                'Coefficient': 'Coefficient (uni)',
                'LowerCI': 'LowerCI (uni)',
                'UpperCI': 'UpperCI (uni)',
                'p-value': 'p-value (uni)'
            }, inplace=True)
    elif reg_type.lower() == 'multi':
        if model_type.lower() == 'logistic':
            summary_df.rename(columns={
                'OddsRatio': 'OddsRatio (multi)',
                'LowerCI': 'LowerCI (multi)',
                'UpperCI': 'UpperCI (multi)',
                'p-value': 'p-value (multi)'
            }, inplace=True)
        else:
            summary_df.rename(columns={
                'Coefficient': 'Coefficient (multi)',
                'LowerCI': 'LowerCI (multi)',
                'UpperCI': 'UpperCI (multi)',
                'p-value': 'p-value (multi)'
            }, inplace=True)

    # 13. Print results if requested
    if print_results:
        print(summary_df)

    # 14. Return summary
    return summary_df
