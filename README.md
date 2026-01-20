# Pretty View

## Overview
This project entails a .py source file containing two functions.
The purpose of these functions is to enable people who are not python 
savvy to utilise the scikit-learn linear regression module in running 
multiple linear regression.
With these functions all the user has to do is to pass the dataset as an
argument to the function and everything else is taken care of.
The user in return gets as an output the regression coefficients, intercept (constant),
Standard deviation, confidence interval (95%), and metric such as R-Squared, Mean Squared Error (MSE),
Relative Mean Squared Error (RMSE) and also info on the split such as the train and test data.
Because else is carried out in the background by the functions, a user may just need two to three lines of 
code to get the job done.

Below are the respective docstrings to the functions:


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



