import json
import plotly
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

# Define custom_tokenizer
def custom_tokenizer(text, url_place_holder_string="urlplaceholder"):
    lemmatizer = WordNetLemmatizer()

    # Replace all URLs with a placeholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Tokenization using regular expressions
    tokens = regexp_tokenize(text, pattern=r'\s|[\.,;!?()"]+', gaps=True)

    # Lemmatization and preprocessing
    clean_tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens]
    
    return clean_tokens

# Define Word2VecVectorizer class
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
        self.vector_size = model.wv.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([np.mean([self.model.wv[word] for word in words if word in self.model.wv]
                                 or [np.zeros(self.vector_size)], axis=0)
                         for words in X])

# Load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# Load model
model = joblib.load("/Users/RobertHolzmann/Downloads/Disaster/Disaster_Response/models/classifier.joblib")

# Index webpage displays visuals and receives user input text for the model
@app.route('/')
@app.route('/index')
def index():
    # Data for genre distribution bar chart
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Data for correlation heatmap
    classification_labels = df.iloc[:, 4:]  # Assuming columns 4 onwards are classification labels
    correlation_matrix = classification_labels.corr().values
    labels = classification_labels.columns

    # Data for genre vs. classification count stacked bar chart
    genre_classification_counts = df.groupby('genre').sum()[classification_labels.columns]

    # Create bar chart for genre distribution
    graphs = [
        # Genre Distribution Bar Chart
        {
            'data': [
                Bar(x=genre_names, y=genre_counts)
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        # Stacked Bar Chart for Genre vs. Classification Count
        {
            'data': [
                Bar(
                    x=genre_classification_counts.index,  # Genres on x-axis
                    y=genre_classification_counts[label],  # Count of each classification label
                    name=label
                ) for label in genre_classification_counts.columns
            ],
            'layout': {
                'title': 'Genre vs. Classification Count',
                'barmode': 'stack',
                'xaxis': {'title': 'Genre'},
                'yaxis': {'title': 'Classification Count'},
                'height': 800  # Increased height for better visibility
            }
        },
        # Correlation Heatmap
        {
            'data': [
                Heatmap(
                    z=correlation_matrix,
                    x=labels,
                    y=labels,
                    colorscale='Reds'  # Use red color for the heatmap
                )
            ],
            'layout': {
                'title': 'Heatmap of Classification Labels',
                'xaxis': {
                    'showticklabels': True,
                    'tickangle': -45,  # Rotate x-axis labels to avoid overlap
                },
                'yaxis': {
                    'showticklabels': True
                },
                'height': 900,  # Reduced size to about 3/4 of the previous size
                'width': 1000,   # Adjusting width accordingly
                'autosize': False,
                'margin': {
                    'l': 150,  # Increased left margin for y-axis labels
                    'r': 150,  # Right margin
                    't': 50,   # Top margin
                    'b': 150   # Increased bottom margin for x-axis labels
                }
            }
        }
    ]

    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    query = request.args.get('query', '') 
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template('go.html', query=query, classification_result=classification_results)

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
