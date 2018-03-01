import logging
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import analyzer.utils as utl
import matplotlib.dates as dates
from scipy.optimize import curve_fit
from sklearn import linear_model

log = logging.getLogger()


class Models(object):
    def __init__(self):
        self.model = None

    def set_model(self, model):
        if model == 'Ordinary Least Squares':
            self.model = linear_model.LinearRegression()
        elif model == 'Ridge Regression':
            self.model = linear_model.Ridge()
        elif model == 'Log':
            self.model = self.log_func
        elif model == 'Linear':
            self.model = self.lin_func

    @staticmethod
    def log_func(x, a, b):
        return a * np.log(x) + b

    @staticmethod
    def lin_func(x, a, b):
        return a * x + b

    def get_model(self, model):
        self.set_model(model)
        return self.model

    def get_coefficients(self, x, y, com):
        if len(x) < 2:
            logging.warning('Only one x, y provided.  Could not fit ' +
                            'parameter combination: ' + str(com))
            return [0, 0], [0, 0]
        try:
            popt, pcov = curve_fit(self.model, x, y)
        except RuntimeError:
            logging.warning('Could not fit parameter combination: ' + str(com))
            return [0, 0], [0, 0]
        return popt, pcov


class JobHandler(object):
    def __init__(self):
        self.models = Models()
        self.model = None
        self.model_dict = {}
        self.model_df = pd.DataFrame()
        self.unique_val_dict = {}
        self.combin_key = {}
        self.combin = None
        self.job_name = None

    def apply_model(self, job_name, df, model, target, predictors, parameters,
                    uid, export_csv):
        logging.info('Applying model ' + str(model) + ' for job: ' + job_name)
        self.model_df = pd.DataFrame()
        self.job_name = job_name
        df = self.df_col_to_type(df, target, predictors)
        self.model = self.models.get_model(model)
        self.get_unique_param_combinations(df, parameters)
        self.loop_all_combinations(df, target, predictors, parameters, uid)
        print(df.head())
        self.add_const_to_df(job_name, model, target, predictors, parameters)
        if export_csv:
            self.export_model_df_to_csv(export_csv)

    def loop_all_combinations(self, df, target, predictors, parameters, uid):
        total_combinations = float(len(self.combin))
        percent_threshold = 10.0
        logging.info(str(total_combinations) + ' possible combinations.')
        for idx, com in enumerate(self.combin):
            percent = (float(idx) / total_combinations) * 100.0
            if percent > percent_threshold:
                percent_threshold += 10.0
                logging.info('Completed: ' + str(round(percent)) + '%.')
            self.slice_and_model(df, com, target, predictors, parameters, uid)

    def slice_and_model(self, df, com, target, predictors, parameters, uid):
        df = self.slice_df_for_model(df, com)
        uids = df[uid].drop_duplicates().tolist()
        df = self.group_df(df, target, predictors, parameters)
        df = df[df[target] != 0]
        if not df.empty:
            min_date = df[predictors].min()
            max_date = df[predictors].max()
            popt = self.compute_model(df, predictors, target, com)
            self.add_model_row(com, uids, popt, min_date, max_date)

    def add_model_row(self, com, uids, popt, min_date, max_date):
        model_row = {'Specified Parameters': com,
                     'Unique Identifiers': uids,
                     'modelcoefa': popt[0],
                     'modelcoefb': popt[1],
                     'modelcoefc': popt[2],
                     'min_date': min_date,
                     'max_date': max_date}
        self.model_df = self.model_df.append(model_row, ignore_index=True)

    @staticmethod
    def plot_fig(model, popt, x, y_actual):
        plt.figure()
        plt.plot(x, y_actual, 'bo', label='Actual Data')
        plt.plot(x, model(x, popt[0], popt[1]), 'r-', label="Model Fit")
        plt.legend(loc='best')

    @staticmethod
    def save_fig(filename, file_path='plots/', ext='.jpg'):
        plt.savefig(file_path + filename + ext)
        plt.close()

    def compute_model(self, df, predictors, target, com):
        x = self.get_x_from_date(df, predictors)
        y = self.get_y(df, target)
        popt, pcov = self.models.get_coefficients(x, y, com)
        popt = np.append(popt, [len(x)])
        self.plot_fig(self.models.model, popt, x, y)
        self.save_fig(filename=str(self.job_name) + '_' + str(com.items()))
        return popt

    def get_unique_param_combinations(self, df, parameters):
        unique_df = df[parameters].drop_duplicates()
        self.combin = unique_df.to_dict(orient='records')

    def add_const_to_df(self, job_name, model, target, predictors, parameters):
        self.model_df['job_name'] = str(job_name)
        self.model_df['target'] = str(target)
        self.model_df['modeltype'] = str(model)
        self.model_df['predictors'] = str(predictors)
        self.model_df['parameters'] = str(parameters)
        self.model_df['modelname'] = \
            (self.model_df['job_name'].astype(str) + ' | ' +
             self.model_df['modeltype'].astype(str) + ' | ' +
             self.model_df['target'].astype(str) + ' | ' +
             self.model_df['predictors'].astype(str) + ' | ' +
             self.model_df['Specified Parameters'].astype(str))

    def export_model_df_to_csv(self, filename):
        logging.info('Exporting job to: ' + str(filename))
        try:
            self.model_df.to_csv(filename, index=False)
        except IOError:
            filename = filename[:-4] + '_temp' + filename[-4:]
            logging.warning('Could not export. Retrying as: ' + str(filename))
            self.export_model_df_to_csv(filename)

    def export_model_df_to_cb(self, ):
        pass

    @staticmethod
    def slice_df_for_model(df, com):
        for col, val in com.items():
            df = df[df[col] == val]
        return df

    @staticmethod
    def group_df(df, target, predictors, parameters):
        col_names = [predictors] + parameters
        df = df[col_names + [target]]
        df = df.groupby(col_names)[target].sum().reset_index()
        df[target] = df[target].cumsum()
        return df

    @staticmethod
    def get_x_from_date(df, col):
        x = matplotlib.dates.date2num(df[col])
        x = x - x.min() + 1.0
        return x

    @staticmethod
    def get_y(df, target):
        y = df[target]
        return y

    @staticmethod
    def df_col_to_type(df, target, predictors):
        df = utl.df_col_to_type(df, predictors, 'DATE')
        df = utl.df_col_to_type(df, target, 'REAL')
        return df
