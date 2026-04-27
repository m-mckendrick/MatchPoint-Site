# Project Log

Please regularly update this file to record your project progress. You should be updating the project log _at least_ once a fortnight.
this log is basically a copy of the commit history will be more in depth later

## Week 1 [19.01.2026]

- Set up IntelliJ project and Python virtual environment.

- Began initial dataset cleaning using ATP match data.

- Started learning about tennis‑specific data structures and cleaning approaches.

- Used online tutorials to understand preprocessing workflows (e.g., YouTube tutorial referenced).

## Week 2 [26.01.2026]
- Added serve and return features and implemented both black‑box and white‑box tests.

- Added tiebreak extraction logic and validated it with new tests.

- Introduced match context features and updated tests accordingly.

- Reflection: early testing revealed several extraction issues which were fixed before they could propagate.

## Week 3 [w/c 02.02.2026]

- Added feature_builder and created pytest fixtures for modular testing.

- Added “is retirement” feature after identifying retirement cases in the dataset.

- Normalised missing values and improved cleaning consistency.

- Added convert_to_p1p2.py and mirror_rows.py to prevent model bias.

- Reflection: this week significantly improved fairness and consistency of the dataset.

## Week 4 [w/c 09.02.2026]

- Integrated p1/p2 conversion and mirroring into the cleaning pipeline.

- Added value validation and removed non‑ATP events.

- Reorganised tests into a clearer structure.

- Reflection: cleaning pipeline reached a stable, reproducible state.

## Week 5 [w/c 16.02.2026]

- Added dataset preparation module for modular data prep.

- Added additional tests for match context and feature correctness.

- Fixed issues with new feature columns discovered during testing.

- Reflection: testing continued to catch subtle issues early.

## Week 6 [w/c 23.02.2026]

- Began training baseline logistic regression model.

- Fixed scaler implementation and removed unavailable stats from training.

- Added feature validation checks and improved save paths for cleaned CSVs.

- Reflection: first full end‑to‑end training run completed successfully.

## Week 7 [w/c 02.03.2026]

- Added random forest model for comparison.

- Reorganised directories to match new architecture.

- Added tests for player and tournament info expected in the final output.

- Reflection: started planning how explainability and UI would integrate with backend outputs.

## Week 8 [w/c 09.03.2026]

- Added tests for explainability pipeline and integrated new fixtures.

- Refined extraction logic for tiebreaks and datetime fields.

- Continued improving feature builder modularity.

- Reflection: groundwork laid for prediction‑time explainability.

## Week 9 [w/c 16.03.2026]

- Fixed model parameter entry issues and improved test coverage.

- Refined match context and feature extraction after running full pytest suite.

- Reflection: stabilised the feature pipeline before moving to API development.

## Week 10 [w/c 23.03.2026]

- No major commits — planning and reviewing architecture for API integration.

- Reflection: focused on preparing for prototype development.

## Week 11 [w/c 30.03.2026]

- No major commits — continued planning for API and frontend prototype.

## Week 12 [w/c 06.04.2026]

- Built FastAPI backend and created simple index.html and results.html.

- Added JS form submission logic and initial results rendering.

- Moved narrative outside widget box and fixed progress bar fill logic.

- Reflection: first working prototype of full prediction → explanation → UI pipeline.

## Week 13 [w/c 13.04.2026]

- Merged feature/api into dev and tagged v0.1‑prototype.

- Streamlined metadata retrieval and improved API structure.

- Added decoupled JSON structure to allow backend evolution without breaking frontend.

- Tagged v0.2‑prototype.

- Renamed loaders to avoid confusion and fixed import issues.

-  Reflection: prototype validated and ready for final modular data prep and feature engineering.


