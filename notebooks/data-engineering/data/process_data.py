import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Reads messages and categories into dataframes.

    Returns:
        Dataframe containing merged messages and categories data.
    '''
    messages = pd.read_csv(filepath_or_buffer=messages_filepath)
    categories = pd.read_csv(filepath_or_buffer=categories_filepath)

    messages.drop_duplicates(inplace=True)
    categories.drop_duplicates(inplace=True)

    # merge datasets
    df = messages.merge(categories, how="inner", on="id")

    categories = categories.categories.str.split(";", expand=True)
    row = categories.iloc[0,:]

    def extract_name(column_name):
        return column_name[0:-2]

    category_colnames = row.apply(extract_name)

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop("categories", axis=1, inplace=True)

    df = pd.concat([df.reset_index(drop=True), categories.reset_index(drop=True)], axis=1)

    # Maximum value for 'related' column is '2'. Fix by replacing 2 by 1 for now. We can update later.
    df['related'] = df['related'].map(lambda x: 1 if x==2 else x)

    return df

def save_data(df, database_filename):
    '''
    Saves dataframe to database.

    Returns:
        Nothing.
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('model_data5', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

#        print('Cleaning data...')
#        df = clean_data(df)
        
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
#%%

#%%
