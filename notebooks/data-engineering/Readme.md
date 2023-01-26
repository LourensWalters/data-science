
# Overview
The purpose of this project is to build machine learning pipeline, which consists of Natural Language Processing model and Classification model, to categorize disaster response messages appropriately according to which disaster relief agency the message should be sent to. The project includes a web app which an emergency worker can use to input a message to obtain a classification indicating where the message should be sent to.

# Components
The project is divided into three components:

1. **ETL Pipeline:** Load datasets, clean the data and store it in a SQLite database
2. **ML Pipeline:** Build a text processing and machine learning pipeline, train a model to classify text message in categories
3. **Flask Web App:** Show model results in real time

# Requirements
- Python 3.10
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy

# Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster.db`
    - To run ML pipeline that trains classifier and saves
      `python models/train_classifier.py data/disaster.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
   `python run.py`

3. Go to http://0.0.0.0:3001/

# Classification Model
The dataset is imbalanced (ie some labels like water have few examples) which contributes to lower score for the ones which have less data. That is why the overall score is low. For skewed datasets, accuracy as a metric for model evaluation is not preferred. In this case, it's better to have more FP than FN, as it makes sure than the messages are at least sent to appropriate disaster relief agency, instead of ignoring some really urgent messages (FN). Therefore, it is important to have a classification model that shows low number of FN -> high recall value.

<p align="center">
  <img src="/images/accuracy.png" height="600" width="800" />
</p>

# Acknowledgements
- [Udacity](https://www.udacity.com) for providing such an interesting and meaningful project
- [Figure Eight](https://appen.com) for providing real-world dataset

# Results
1. Input message to get classification results
<p align="center">
  <img src="/images/question.png" height="200" width="800" />
</p>

2. Example: The categories which the message belongs to highlighted in green
<p align="center">
  <img src="/images/result.png" height="600" width="800" />
</p>

3. Overview of Training Dataset

<p align="center">
  <img src="/images/categories.png" height="400" width="800" />
</p>
