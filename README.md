# Disaster Response Pipeline Project

### Link To Github Repo

[Udacity Disaster Response Project](https://github.com/lebrontaso/Udacity-Disaster-Response-Project)

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python run.py`

3. Go to http://localhost:3001/

### Python Package Requirements

Python3.6 has been used and package requirements for this project are:

-   numpy
-   pandas
-   matplotlib
-   plotly
-   scikit-learn
-   nltk
-   flask
-   sqlalchemy
