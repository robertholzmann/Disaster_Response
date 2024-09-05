# Disaster Response Pipeline Project

This project is a part of Udacity's Data Scientist Nanodegree Program in collaboration with Figure Eight. It aims to build a Natural Language Processing (NLP) tool to categorize disaster-related messages in real-time.

## Table of Contents

1. [Introduction](#introduction)
2. [File Descriptions](#file-descriptions)
3. [Installation](#installation)
4. [Instructions](#instructions)
5. [Acknowledgements](#acknowledgements)
6. [Screenshots](#screenshots)

## Introduction

The Disaster Response Pipeline Project uses pre-labeled disaster messages to build a model that can classify messages received during disaster events. The goal is to route these messages to the appropriate disaster response agency efficiently.

The project includes:
- **ETL Pipeline**: Extracts data, cleans it, and stores it in a SQLite database.
- **ML Pipeline**: Trains a machine learning model to classify messages.
- **Web Application**: Provides a user interface for classifying messages and visualizing data.

## File Descriptions

### Folder: `app`
- **`run.py`**: Python script to launch the Flask web application.
- **`templates/`**: Contains HTML templates (`go.html` and `master.html`) for the web application.

### Folder: `data`
- **`disaster_messages.csv`**: Messages sent during disaster events (provided by Figure Eight).
- **`disaster_categories.csv`**: Categories associated with the messages.
- **`process_data.py`**: ETL script to clean and store data in a SQLite database.
- **`DisasterResponse.db`**: SQLite database containing cleaned data.

### Folder: `models`
- **`train_classifier.py`**: ML pipeline script to train and save the classification model.
- **`classifier.pkl`**: Pickle file containing the trained model.
- **`ML Pipeline Preparation.ipynb`**: Jupyter Notebook for understanding and tuning the ML pipeline.

## Installation

Ensure you have Python 3.5 or higher and the Anaconda distribution.

### Dependencies

Install the following Python libraries:

```bash
pip install SQLAlchemy nltk pandas numpy scipy scikit-learn flask plotly
