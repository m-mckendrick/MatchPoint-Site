# MatchPoint-Tennis Match Prediction System

## Information about this repository

This repository contains the full development workflow for MatchPoint, a machine‑learning project designed to predict tennis match outcomes using historical player performance data. The project is structured using a clean, industry‑standard Git branching model to keep development organised, traceable, and easy to review.


## How to run the site
# the data pipeline
first - clean all the raw files by navigating to the clean_pipeline script
        and run it (make sure all files run from the project root) 

second - navigate to the feature_builder file and run it this may take several hours 
        as its adding features to all the atp matches from 2000-2023

third - once that's done the model can be trained. this time run the xgboost_model file
        and also make sure to run teh shap_modular file this creates and saves 
        the shap explainer
        

# entering the site
first - activate the virtual environment (python 3.11.9) 
        in the terminal inside teh project root using  
        .\.venv\Scripts\activate

second - install all dependancies in the requirements txt file 
        using pip install -r requirements.txt

third - run the app using 
        uvicorn src.api.main:app --reload
        then you should see the url of teh site copy it into you browser 
        and you'll be on teh first page

 



