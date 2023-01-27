import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
nltk.download("averaged_perceptron_tagger")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score

def load_data(database_filepath = 'sqlite:///data/disaster.db'):
    '''
    Reads data into dataframes from database.

    Returns:
        Dataframes containing dependent and independent variables, as well column names for independent variables.
    '''
    engine = create_engine(database_filepath)
    sql = "select * from model_data4"
    df = pd.read_sql(sql, con=engine)

    # We remove 'child_alone' column as it has no values.
    df = df.drop(['child_alone', 'offer', 'tools'], axis = 1)

    X = df[["message", "genre"]]
    Y = df.iloc[:,5:]

    column_names = list(Y.columns.values)

    return X, Y, column_names

def tokenize(text):
    '''
    Extracts tokens from string by means of cleaning, tokenising and lemmatising input.

    Input:
        Text string to be tokenised.

    Output:
        List of cleaned tokens.
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # detect all URL present in the messages
    detected_urls = re.findall(url_regex, text)
    # replace URL with "urlplaceholder"
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# Engineer new features.
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Function to extract starting verb from each sentence.

    Input:
        Base estimator and
        Transformer classes used within ML pipelines.
    Return:
        Transform method returns a dataframe containing transformed values.
    '''

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def pipeline2(clf  = AdaBoostClassifier(random_state = 1)):
     pipeline = Pipeline([
         ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
         ])),
         ('clf', MultiOutputClassifier(clf))
     ])
     return pipeline

def build_model():
    '''
    Builds the model to be fitted to the data.

    Input:
        None.

    Output:
        Returns the model object which could be used to fit and predict on data.
    '''
    parameters_ada = {
    'clf__estimator__learning_rate': [0.1, 0.3],
    'clf__estimator__n_estimators': [100, 200]
    }

    pipeline_2 = pipeline2()
    pipeline_cv = GridSearchCV(estimator=pipeline_2, param_grid=parameters_ada, cv=3, verbose=10, n_jobs=-1)

    #pipeline_cv = GridSearchCV(estimator=pipeline_2, param_grid=parameters_ada, cv=3, verbose=10, n_jobs=-1)

    return pipeline_cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function that predicts values based on model selected, and generates an accuracy report.

    Input:
        Model:      Model to be used for predictions.
        X_test:     Test data containing independent variables to be used for predictions.
        Y_test:     Dependent variable to be used for accuracy evaluation.

    Returns:
        Nothing.
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names = category_names))

def save_model(model, model_filepath):
    '''
    Saves model to pickle file.

    Input:
        model:          Model to be saved.
        model_path:     Path to where model should be saved.

    Returns:
        Nothing.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data("sqlite:///" + database_filepath)
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        X_load = X["message"].values
        Y_load = Y.values
        X_train, X_test, Y_train, Y_test = train_test_split(X_load, Y_load, random_state = 1)
        print("Xtrain size: {}".format(X_train.shape[0]))
        print("Ytrain size: {}".format(Y_train.shape[0]))

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        np.random.seed(333)


        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
#%%
