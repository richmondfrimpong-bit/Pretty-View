from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


def pv_linear_regression(dataset):
    """
    Parameters
    __________
    dataset : This input should be an in-built dataset.

                    This is the input data

    Returns
    _______
    An output containing the regression coefficients of variables, intercept(constant term),standard deviation, confidence interval, R-squared, MSE, RMSE


    Example
    _______

    from sklearn.datasets import fetch_california_housing


    from pretty_view import pv_linear_regression


    data = fetch_california_housing()


    regress = pv_linear_regression(data)


    print(regress)
    """
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target,random_state=78)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X=X_test)
    expected = y_test
    datFR = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    ray = []
    for item in datFR.columns:
        if item not in dataset.target_names:
            ray.append(round(float(datFR.get(item).std()),6))
    conf_low = []
    conf_upper = []
    for combo in zip(model.coef_, ray):
        esti, stad_dev = combo
        conf_low.append(round(float(esti - 1.96 * (stad_dev / np.sqrt(len(dataset.target)))), 6))
        conf_upper.append(round(float(esti + 1.96 * (stad_dev / np.sqrt(len(dataset.target)))), 6))
    dict = {'Variable':[x for x in dataset.feature_names],
            'Estimate': [y for y in model.coef_],
            'Std': [st for st in ray],
            'Conf_lower(0.05)': [conf for conf in conf_low],
            'Conf_upper(0.05)': [confu for confu in conf_upper]
            }
    par_datF = pd.DataFrame(dict, columns=['Variable', 'Estimate', 'Std',
                                           'Conf_lower(0.05)', 'Conf_upper(0.05)'])
    return (print(f'==========================================\n' +
     f'|| Shape of train data: {X_train.shape}      ||\n' +
     f'==========================================\n' +
     f'|| Shape of test data: {X_test.shape}        ||\n' +
     f'==========================================\n' +
     f'|| R-Squared: {r2_score(y_true=expected, y_pred=predicted)}        ||\n'  +
     f'==========================================\n' +
     f'|| MSE: {mean_squared_error(y_true=expected, y_pred=predicted)}              ||\n' +
     f'==========================================\n' +
     f'|| RMSE: {np.sqrt(mean_squared_error(y_true=expected, y_pred=predicted))}              ||\n' +
     f'=========================================='),
        print(par_datF),
        print(f'Intercept:  {model.intercept_}'))


def pv_linear_reg_datf(dataframe, dependent):
    """
        Parameters
        __________
        dataset : This could be a dataframe or and imported csv
                        This is the input data
        dependent: The dependent variable to be used in the regression
                    This should be the name of the column entered as a string
                    The name entered should match that in the actual dataset


        Returns
        _______
        An output containing the regression coefficients of variables, intercept(constant term),
         stdndard deviation, confidence interval, R-squared, MSE, RMSE



        Example
        _______
        e.g.1   from pretty_view import pv_linear_reg_datf
                import pandas as pd

                data = pd.read_csv('your file directory')

                regress = pv_linear_regression(data, 'name_of_dependent_variable')

                print(regress)

        e.g.2  from pretty_view import pv_linear_reg_datf

               dict = {'machine_1': [45, 68, 78], 'machine_2': [52, 85, 79],
                'average_score': [78, 19, 98]}

               data =  pd.DataFrame(dict,columns=['machine_1',
               'machine_2', 'average_score'])

               regress = pv_linear_reg_datf(data, dependent='average_score')

               print(regress)

    """

    independent = pd.DataFrame(dataframe)
    independent = independent.drop(dependent, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(independent, dataframe.get(dependent).values, random_state=78)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X=X_test)
    expected = y_test
    ray = []
    for item in dataframe.columns:
        if item in independent.columns:
            ray.append(round(float(dataframe.get(item).std()),6))
    conf_low = []
    conf_upper = []
    for combo in zip(model.coef_, ray):
        esti, stad_dev = combo
        conf_low.append(round(float(esti - 1.96 * (stad_dev / np.sqrt(len(dataframe.get(dependent))))), 6))
        conf_upper.append(round(float(esti + 1.96 * (stad_dev / np.sqrt(len(dataframe.get(dependent))))), 6))
    diction_ary = {'Variable': [x for x in independent.columns],
                'Estimate': [y for y in model.coef_],
                   'std': [st for st in ray],
                   'Conf_lower(0.05)': [conf for conf in conf_low],
                   'Conf_upper(0.05)': [confu for confu in conf_upper]}
    par_datF = pd.DataFrame(diction_ary, columns=['Variable', 'Estimate', 'std',
                                                  'Conf_lower(0.05)', 'Conf_upper(0.05)'])
    return (print(f'==========================================\n'
          f'|| Shape of train data: {X_train.shape}         ||\n' +
          f'==========================================\n' +
          f'|| Shape of test data: {X_test.shape}          ||\n' +
          f'==========================================\n' +
          f'|| R-Squared: {r2_score(y_true=expected, y_pred=predicted)}     ||\n' +
          f'==========================================\n'
          f'|| MSE: {mean_squared_error(y_true=expected, y_pred=predicted)}               ||\n' +
          f'==========================================\n' +
          f'|| RMSE: {np.sqrt(mean_squared_error(y_true=expected, y_pred=predicted))}               ||\n' +
          f'=========================================='),
            print(par_datF),
            print(f'Intercept  {model.intercept_}'))






