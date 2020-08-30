import nltk
import down_nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score
from sqlalchemy import create_engine
import pandas as pd
import sys
from sklearn.externals import joblib


def load_data(database_filepath):
    """Load data from Sqlite Database

    Args:
        database_filepath (str): Sqlite Database Filename

    Returns:
        numpy.ndArray: Messages for Disaster Responses
        numpy.ndArray: Categories for Disaster Responses
        list: Category Names
    """
    engine = create_engine('sqlite:///{0}'.format(database_filepath))
    df = pd.read_sql_table('DisasterCleaned', engine)
    df = df.dropna(subset=['related'])
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns
    return X, Y, category_names


def tokenize(text):
    """Clean, Tokenize sentences and Lemmatize words
    Args:
        text (str): Raw Text

    Returns:
        list: Tokenized and Cleaned sentence
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Pipeline Classification Model

    Returns:
        scikit-learn.GridSearchCV: Classification Pipeline Model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2))
        # 'clf__estimator__n_estimators': [50, 100],
        # 'clf__estimator__min_samples_split': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluation of the Classification model

    Args:
        model (scikit-learn.GridSearchCV): Classification Pipeline Model
        X_test (numpy.ndArray): X_Test
        Y_test (numpy.ndArray): Y_Test 
        category_names (list): list of category names
    """
    y_pred = model.predict(X_test)
    for i in range(y_pred.shape[1]):
        rc_score = recall_score(Y_test[:, i], y_pred[:, i], average='macro')
        ac_score = accuracy_score(Y_test[:, i], y_pred[:, i])
        pc_score = precision_score(Y_test[:, i], y_pred[:, i], average='macro')
        print(category_names[i])
        print("\tAccuracy: {0:4f}\t% Precision: {1:4f}\t% Recall: {2:4f}".format(
            ac_score, pc_score, rc_score))


def save_model(model, model_filepath):
    """Save Model

    Args:
        model (scikit-learn.GridSearchCV): Classification Pipeline Model
        model_filepath (str): Filepath to save model
    """
    joblib.dump(model.best_estimator_, model_filepath)


def main():
    """
        Main code to run Machine Learning Pipeline of Disaster Response Project
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
