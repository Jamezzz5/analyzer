import sys
import logging
import json
from io import BytesIO
import pandas as pd
import analyzer.models as mod
import analyzer.utils as utl
from sqlalchemy import create_engine

log = logging.getLogger()

job_name = 'Job Name'
source = 'Source'
con_file = 'Configuration'
raw_file = 'Raw'
target = 'Target'
predictors = 'Predictors'
model = 'Model'
parameters = 'Parameters'
unique_id = 'Unique Identifier'
export_csv = 'Export CSV'


class Config(object):
    def __init__(self, config_path='config/', config_file='config.csv'):
        logging.info('Initiating config_file: ' + str(config_file))
        self.path = config_path
        self.file = self.path + config_file
        self.raw_config = pd.read_csv(self.file)
        for col in [raw_file, con_file, source]:
            self.raw_config = self.raw_config.sort_values(by=col)
        self.keys = self.raw_config[job_name].tolist()
        self.config = self.raw_config.set_index(job_name).to_dict()
        for col in [parameters]:
            self.config[col] = ({key: list(value.split('|')) for key, value in
                                 self.config[col].items()})
        self.job = None
        self.df = None
        self.cls = None
        self.raw_source = None
        self.last_raw_source = None

    def loop(self):
        jh = mod.JobHandler()
        utl.dir_check('plots')
        for key in self.keys:
            self.do_job(jh, key)

    def do_job(self, jh, key):
        self.set_job_and_get_df(key)
        jh.apply_model(key, self.df, self.job[model], self.job[target],
                       self.job[predictors], self.job[parameters],
                       self.job[unique_id], self.job[export_csv])
        self.last_raw_source = self.raw_source

    def set_job_and_get_df(self, key):
        logging.info('Getting dataframe for job: ' + str(key))
        self.set_job(key)
        if self.raw_source != self.last_raw_source:
            self.get_job_class(self.job[source], self.job[con_file])
            self.get_raw_df(self.job[raw_file])

    def set_job(self, parent_key):
        self.job = {key: value[parent_key]
                    for key, value in self.config.items()}
        self.raw_source = [self.job[source], self.job[con_file],
                           self.job[raw_file]]

    def get_job_class(self, data_source, config_file):
        if data_source == 'csv':
            self.cls = Csv()
        elif data_source == 'db':
            self.cls = DB(config=config_file)

    def get_raw_df(self, filename):
        logging.info('Getting data from: ' + str(filename))
        self.df = self.cls.load_raw_df(filename)


class Csv(object):
    def __init__(self):
        self.file = None

    @staticmethod
    def load_raw_df(file, file_path='raw/'):
        full_file = file_path + file
        try:
            df = pd.read_csv(full_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(full_file, encoding='iso-8859-1')
        return df


class DB(object):
    def __init__(self, config, config_path='config/'):
        self.config = config
        self.config_path = config_path
        self.user = None
        self.pw = None
        self.host = None
        self.port = None
        self.db = None
        self.config_file = self.config_path + self.config
        self.config_list = []
        self.engine = None
        self.connection = None
        self.cursor = None
        self.output = None
        self.input_config(self.config_file)
        self.conn_string = ('postgresql://{0}:{1}@{2}:{3}/{4}'.
                            format(*self.config_list))

    def input_config(self, config_file):
        logging.info('Loading DB config file: ' + str(config_file))
        self.load_config()
        self.check_config()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except IOError:
            logging.error(self.config_file + ' not found.  Aborting.')
            sys.exit(0)
        self.user = self.config['USER']
        self.pw = self.config['PASS']
        self.host = self.config['HOST']
        self.port = self.config['PORT']
        self.db = self.config['DATABASE']
        self.config_list = [self.user, self.pw, self.host, self.port, self.db]

    def check_config(self):
        for item in self.config_list:
            if item == '':
                logging.warning(item + 'not in DB config file.  Aborting.')
                sys.exit(0)

    def connect(self):
        logging.debug('Connecting to DB at Host: ' + self.host)
        self.engine = create_engine(self.conn_string)
        self.connection = self.engine.raw_connection()
        self.cursor = self.connection.cursor()

    def load_raw_df(self, table):
        columns = self.get_column_names(table)
        self.connect()
        command = """
                  SELECT * FROM {0}.{1}
                  """.format(self.db, table)
        self.cursor.execute(command)
        data = self.cursor.fetchall()
        data = pd.DataFrame(data=data, columns=columns)
        return data

    def get_column_names(self, table):
        self.connect()
        command = """
                  SELECT *
                  FROM information_schema.columns
                  WHERE table_schema = '{0}'
                  AND table_name = '{1}'
                  """.format(self.db, table)
        self.cursor.execute(command)
        columns = self.cursor.fetchall()
        columns = [x[3] for x in columns]
        return columns

    def read_rds(self, table, select_col, where_col, where_val):
        self.connect()
        if select_col == where_col:
            command = """
                      SELECT {0}.{1}.{2}
                       FROM {0}.{1}
                       WHERE {0}.{1}.{3} IN ({4})
                      """.format(self.db, table, select_col, where_col,
                                 ', '.join(['%s'] * len(where_val)))
        else:
            command = """
                      SELECT {0}.{1}.{2}, {0}.{1}.{3}
                       FROM {0}.{1}
                       WHERE {0}.{1}.{3} IN ({4})
                      """.format(self.db, table, select_col, where_col,
                                 ', '.join(['%s'] * len(where_val)))
        self.cursor.execute(command, where_val)
        data = self.cursor.fetchall()
        if select_col == where_col:
            data = pd.DataFrame(data=data, columns=[select_col])
        else:
            data = pd.DataFrame(data=data, columns=[select_col, where_col])
        return data

    def read_rds_table(self, table, where_col, where_val):
        self.connect()
        command = """
                  SELECT *
                   FROM {0}.{1}
                   WHERE {0}.{1}.{2} IN ({3})
                  """.format(self.db, table, where_col,
                             ', '.join(['%s'] * len(where_val)))
        self.cursor.execute(command, where_val)
        data = self.cursor.fetchall()
        self.connect()
        command = """
                  SELECT *
                  FROM information_schema.columns
                  WHERE table_schema = '{0}'
                  AND table_name = '{1}'
                  """.format(self.db, table)
        self.cursor.execute(command)
        columns = self.cursor.fetchall()
        columns = [x[3] for x in columns]
        data = pd.DataFrame(data=data, columns=columns)
        return data

    def update_rows(self, table, set_cols, set_vals, where_col):
        logging.info('Updating ' + str(len(set_vals)) +
                     ' row(s) from ' + table)
        self.connect()
        command = """
                  UPDATE {0}.{1} AS t
                   SET {2}
                   FROM (VALUES {3})
                   AS c({4})
                   WHERE c.{5} = t.{5}
                  """.format(self.db, table,
                             (', '.join(x + ' = c.' + x
                              for x in [where_col] + set_cols)),
                             ', '.join(['%s'] * len(set_vals)),
                             ', '.join([where_col] + set_cols),
                             where_col)
        self.cursor.execute(command, set_vals)
        self.connection.commit()

    def df_to_output(self, df):
        self.output = BytesIO()
        df.to_csv(self.output, sep='\t', header=False, index=False,
                  encoding='utf-8')
        self.output.seek(0)

    def copy_from(self, table, df, columns):
        table_path = self.db + '.' + table
        self.connect()
        logging.info('Writing ' + str(len(df)) + ' row(s) to ' + table)
        self.df_to_output(df)
        cur = self.connection.cursor()
        cur.copy_from(self.output, table=table_path, columns=columns)
        self.connection.commit()
        cur.close()

    def insert_rds(self, table, columns, values, return_col):
        self.connect()
        command = """
                  INSERT INTO {0}.{1} ({2})
                   VALUES ({3})
                   RETURNING ({4})
                  """.format(self.db, table, ', '.join(columns),
                             ', '.join(['%s'] * len(values)), return_col)
        self.cursor.execute(command, values)
        self.connection.commit()
        data = self.cursor.fetchall()
        data = pd.DataFrame(data=data, columns=[return_col])
        return data

    def delete_rows(self, table, where_col, where_val,
                    where_col2, where_vals2):
        logging.info('Deleting ' + str(len(where_vals2)) +
                     ' row(s) from ' + table)
        self.connect()
        command = """
                  DELETE FROM {0}.{1}
                   WHERE {0}.{1}.{2} IN ({3})
                   AND {0}.{1}.{4} IN ({5})
                  """.format(self.db, table, where_col, where_val, where_col2,
                             ', '.join(['%s'] * len(where_vals2)))
        self.cursor.execute(command, where_vals2)
        self.connection.commit()


class DBUpload(object):
    def __init__(self):
        self.db = None
        self.dbs = None
        self.dft = None
        self.table = None
        self.id_col = None
        self.name = None
        self.values = None

    def upload_to_db(self, db_file, schema_file, trans_file, df):
        self.db = DB(db_file)
        logging.info('Uploading  to ' + self.db.db)
        self.dbs = DBSchema(schema_file, trans_file)
        for table in self.dbs.table_list:
            self.upload_table_to_db(table, df)
        logging.info('Successfully upload to ' + self.db.db)

    def upload_table_to_db(self, table, df):
        logging.info('Uploading table ' + table + ' to ' + self.db.db)
        cols = self.dbs.get_cols_for_export(table)
        df = self.dft.slice_for_upload(df, cols)
        df = self.add_ids_to_df(self.dbs.fk, df)
        self.dbs.set_table(table)
        pk_config = {table: list(self.dbs.pk.items())[0]}
        self.set_id_info(table, pk_config, df)
        where_col = self.name
        where_val = self.values
        df_rds = self.read_rds_table(table, list(df.columns),
                                     where_col, where_val)
        df = pd.merge(df_rds, df, how='outer', on=self.name, indicator=True)
        df = df.drop_duplicates(self.name).reset_index()
        self.update_rows(df, df_rds.columns, table)
        self.delete_rows(df, table)
        self.insert_rows(df, table)

    def update_rows(self, df, cols, table):
        df_update = df[df['_merge'] == 'both']
        updated_index = []
        set_cols = [x for x in cols if x not in [self.name, self.id_col]]
        for col in set_cols:
            df_changed = (df_update[df_update[col + '_y'] !=
                                    df_update[col + '_x']]
                          [[self.name, col + '_y']])
            updated_index.extend(df_changed.index)
        if updated_index:
            df_update = utl.get_right_df(df_update, self.name)
            df_update = df_update.loc[updated_index]
            df_update = df_update[[self.name] + set_cols]
            set_vals = [tuple(x) for x in df_update.values]
            self.db.update_rows(table, set_cols, set_vals, self.name)

    def delete_rows(self, df, table):
        df_delete = df[df['_merge'] == 'left_only']
        delete_vals = df_delete[self.name].tolist()
        if delete_vals:
            self.db.delete_rows(table, self.name, delete_vals)

    def insert_rows(self, df, table):
        df_insert = df[df['_merge'] == 'right_only']
        df_insert = utl.get_right_df(df_insert, self.name)
        for fk_table in self.dbs.fk:
            col = self.dbs.fk[fk_table][0]
            if col in df_insert.columns:
                df_insert = utl.df_col_to_type(df_insert, col, 'INT')
        if self.id_col in df_insert.columns:
            df_insert = df_insert.drop([self.id_col], axis=1)
        if not df_insert.empty:
            self.db.copy_from(table, df_insert, df_insert.columns)

    def read_rds_table(self, table, cols, where_col, where_val):
        df_rds = self.db.read_rds_table(table, where_col, where_val)
        df_rds = df_rds[cols]
        df_rds = self.dft.clean_types_for_upload(df_rds)
        return df_rds

    def add_ids_to_df(self, id_config, sliced_df):
        for id_table in id_config:
            df_rds = self.format_and_read_rds(id_table, id_config, sliced_df)
            sliced_df = sliced_df.merge(df_rds, how='outer', on=self.name)
            sliced_df = sliced_df.drop(self.name, axis=1)
            sliced_df = utl.df_col_to_type(sliced_df, self.id_col, 'INT')
        return sliced_df

    def format_and_read_rds(self, table, id_config, sliced_df):
        self.set_id_info(table, id_config, sliced_df)
        self.dbs.set_table(table)
        df_rds = self.db.read_rds(table, self.id_col, self.name, self.values)
        return df_rds

    def set_id_info(self, table, id_config, sliced_df):
        self.table = table
        self.id_col = id_config[table][0]
        self.name = id_config[table][1]
        self.values = sliced_df[self.name].tolist()


class DBSchema(object):
    def __init__(self, config_path='config/', config_file='dbschema.csv'):
        self.config_file = config_file
        self.full_config_file = config_path + self.config_file
        self.table = 'Table'
        self.pk = 'PK'
        self.columns = 'Columns'
        self.fk = 'FK'
        self.split_columns = [self.pk, self.columns, self.fk]
        self.dirty_columns = {self.pk: ':', self.columns: ' ', self.fk: ':'}
        self.table_list = None
        self.config = None
        self.pk = None
        self.cols = None
        self.fk = None
        self.load_config(self.full_config_file)

    def load_config(self, config_file):
        df = pd.read_csv(config_file)
        self.table_list = df[self.table].tolist()
        self.config = df.set_index(self.table).to_dict()
        for col in self.split_columns:
            self.config[col] = {key: list(str(value).split(',')) for
                                key, value in self.config[col].items()}
        for table in self.table_list:
            for col in self.dirty_columns:
                clean_dict = self.clean_table_item(table, col,
                                                   self.dirty_columns[col])
                self.config[col][table] = clean_dict

    def clean_table_item(self, table, config_col, split_char):
        clean_dict = {}
        for item in self.config[config_col][table]:
            if item == str('nan'):
                continue
            cln_item = item.strip().split(split_char)
            if len(cln_item) == 3:
                clean_dict.update({cln_item[0]: [cln_item[1], cln_item[2]]})
            elif len(cln_item) == 2:
                clean_dict.update({cln_item[0]: cln_item[1]})
        return clean_dict

    def set_table(self, table):
        self.pk = self.config[self.pk][table]
        self.cols = self.config[self.columns][table]
        self.fk = self.config[self.fk][table]

    def get_cols_for_export(self, table):
        self.set_table(table)
        fk_list = [self.fk[x][1] for x in self.fk]
        cols_list = self.cols.keys()
        cols_list = [x for x in cols_list if x not in fk_list]
        return fk_list + cols_list


class DFTranslation(object):
    def __init__(self, config_path='config/',
                 config_file='db_df_translation.csv'):
        self.column_db = 'DB'
        self.column_df = 'DF'
        self.column_type = 'TYPE'
        self.config_file = config_file
        self.full_config_file = config_path + self.config_file
        self.translation = None
        self.db_columns = None
        self.df_columns = None
        self.df = None
        self.sliced_df = None
        self.type_columns = None
        self.translation = None
        self.translation_type = None
        self.text_columns = None
        self.date_columns = None
        self.int_columns = None
        self.real_columns = None
        self.upload_id = None
        self.load_translation(self.full_config_file)

    def load_translation(self, config_file):
        df = pd.read_csv(config_file)
        self.db_columns = df[self.column_db].tolist()
        self.df_columns = df[self.column_df].tolist()
        self.type_columns = df[self.column_type].tolist()
        self.translation = dict(zip(df[self.column_df], df[self.column_db]))
        self.translation_type = dict(zip(df[self.column_db],
                                         df[self.column_type]))
        self.text_columns = [k for k, v in self.translation_type.items()
                             if v == 'TEXT']
        self.date_columns = [k for k, v in self.translation_type.items()
                             if v == 'DATE']
        self.int_columns = [k for k, v in self.translation_type.items()
                            if v == 'INT' or v == 'BIGINT'
                            or v == 'BIGSERIAL']
        self.real_columns = [k for k, v in self.translation_type.items()
                             if v == 'REAL' or v == 'DECIMAL']

    def clean_types_for_upload(self, df):
        for col in df.columns:
            if col not in self.translation_type.keys():
                continue
            data_type = self.translation_type[col]
            df = utl.df_col_to_type(df, col, data_type)
        return df

    @staticmethod
    def slice_for_upload(df, columns):
        exp_cols = [x for x in columns if x in list(df.columns)]
        df = df[[exp_cols]]
        return df
