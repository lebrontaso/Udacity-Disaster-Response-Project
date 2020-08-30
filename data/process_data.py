import sys
import numpy as np
import pandas as pd
import sqlalchemy


def load_data(messages_filepath, categories_filepath):
    """Loads data from two files as one file stores messages other one stores categories for the messages. 
    Returns dataframe merged by these two files for machine learning algorithm

    Args:
        messages_filepath (str): Path of messages.csv file
        categories_filepath (str): Path of categories.csv file

    Returns:
        pandas.DataFrame: Concatenated dataframe of two csv files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    categories = clean_categories(categories)

    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    """Removes duplicate rows and rows that have related other than binary label

    Args:
        df (pandas.DataFrame): Dataframe to be cleaned

    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    df = df[~df.duplicated()]
    df = df[df.related != 2.0]

    return df


def save_data(df, database_filename):
    """Save data to sqlite database

    Args:
        df (pandas.DataFrame): Cleaned Dataset
        database_filename (str): Sqlite Database Filename
    """
    engine = sqlalchemy.create_engine(
        'sqlite:///{0}'.format(database_filename))
    df.to_sql('DisasterCleaned', engine, index=False)


def clean_categories(categories):
    """Clean categories dataset to each column for labels as 1 or 0

    Args:
        categories (pandas.DataFrame): categories dataset

    Returns:
        pandas.DataFrame: cleaned categories dataset
    """
    categories = categories['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda s: s.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype('int')
    return categories


def main():
    """
        Main code to run ETL Pipeline of Disaster Response Project
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
