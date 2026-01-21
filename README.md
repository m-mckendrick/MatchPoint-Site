# MatchPoint-Tennis Match Prediction System

## Information about this repository

This repository contains the full development workflow for MatchPoint, a machine‑learning project designed to predict tennis match outcomes using historical player performance data. The project is structured using a clean, industry‑standard Git branching model to keep development organised, traceable, and easy to review.

### Branching Structure
main – the stable, final version of the project. Only completed and reviewed work is merged here.

dev – the active development branch where all features come together before release.

feature/data-prep – used for cleaning and preparing raw datasets.

feature/feature-engineering – used for creating new predictive features.

feature/model-training – used for training, tuning, and evaluating machine‑learning models.

feature/api-endpoints – used for building backend API routes.

feature/frontend-ui – used for developing the dashboard and user interface.

### Workflow
Start work by switching to the relevant feature branch.

Develop and commit changes locally.

Push the feature branch to GitLab.

Merge into dev once the feature is complete and tested.

After all components are integrated, merge dev into main for the final project version.


