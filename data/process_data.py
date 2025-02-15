import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge them on 'id'.

    Args:
    messages_filepath: str. Filepath for the messages dataset.
    categories_filepath: str. Filepath for the categories dataset.

    Returns:
    df: dataframe. Merged pandas dataframe containing messages and categories.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """
    Clean the data by splitting categories, converting them to binary, and removing duplicates.
    Additional steps include filtering out invalid 'related' values and keeping columns with more than one unique value.

    Args:
    df: dataframe. Merged pandas dataframe containing messages and categories.

    Returns:
    df: dataframe. Cleaned dataframe with categories split and converted to binary.
    """
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # Extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Remove rows where 'related' column has the value 2
    df = df[df['related'] != 2]

    # Keep only columns with more than one unique value
    valid_cols = [col for col in df.columns if df[col].nunique() > 1]
    df = df[valid_cols]

    # Drop duplicates
    df = df.drop_duplicates()
    
    return df



def save_data(df, database_filename):
    """
    Save the clean dataset into an SQLite database.

    Args:
    df: dataframe. Cleaned dataframe containing messages and categories.
    database_filename: str. Filepath for the SQLite database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster', engine, index=False, if_exists='replace')  


def main():
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
