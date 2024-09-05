import sys
import pandas as pd
import re
import nltk
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
from joblib import dump
from nltk.tokenize import regexp_tokenize
from nltk.stem import WordNetLemmatizer
import pickle 

nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from the SQLite database and split it into features and target variables.

    Args:
    database_filepath (str): Path to the SQLite database file.

    Returns:
    tuple: X (features), Y (target variables), and category_names (list of category names).
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * FROM disaster', engine)
    
    # Filter out any columns with only a single value
    valid_cols = [col for col in df.columns if df[col].nunique() > 1]
    df = df[valid_cols]
    df = df[df['related'] != 2]
    
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    """
    Tokenizes the text data using regular expressions and lemmatization.

    Args:
    text (str): The text to tokenize.

    Returns:
    list: A list of clean tokens extracted from the text.
    """
    lemmatizer = WordNetLemmatizer()

    # Replace all URLs with a placeholder string
    url_place_holder_string = "urlplaceholder"
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Tokenization using regexp_tokenize
    tokens = regexp_tokenize(text, pattern=r'\s|[\.,;!?()"]+', gaps=True)

    # Lemmatization and preprocessing
    clean_tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens]
    
    return clean_tokens

def build_model():
    """
    Build a machine learning pipeline and use GridSearchCV to optimize it.

    Returns:
    GridSearchCV: A GridSearchCV object containing the machine learning pipeline.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),  # Custom tokenizer step that preprocesses text data
                ('tfidf_transformer', TfidfTransformer())  # Converts preprocessed text to TF-IDF features
            ]))
        ])),
        ('clf', MultiOutputClassifier(GradientBoostingClassifier()))  # The classifier predicts multiple target variables
    ])

    # Hyperparameters for GridSearchCV
    params = {
        'clf__estimator__learning_rate': [0.01, 0.1, 0.2],
        'clf__estimator__n_estimators': [50, 100, 200]
    }
    
    # Create GridSearchCV object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, scoring='accuracy')
    
    return model_pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model by printing out classification reports for each category.

    Args:
    model: The trained machine learning model.
    X_test (DataFrame): Test features.
    Y_test (DataFrame): True labels for test data.
    category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    
    for i, col in enumerate(category_names):
        print(f"Category: {col}")
        print(classification_report(Y_test[col], Y_pred[:, i], zero_division=1))
        print("\n")

def save_model(model, model_filepath):
    """
    Save the trained model to a file.

    Args:
    model: The trained machine learning model.
    model_filepath (str): Path to the file where the model will be saved.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function that orchestrates the loading of data, model training, evaluation, and saving.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
