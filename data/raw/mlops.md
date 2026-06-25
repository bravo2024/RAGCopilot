# MLOps and Deployment

## ML pipelines

An ML pipeline orchestrates data ingestion, validation, preprocessing, training, evaluation, and deployment. Tools like Kubeflow, Airflow, and ZenML manage orchestration.

## Model monitoring

Model monitoring tracks prediction distributions, feature drift, and performance metrics in production. Alerts trigger investigation or retraining when drift exceeds thresholds.

## A/B testing for ML

A/B testing for ML compares a new model against the current production model on live traffic. Metrics like conversion rate and latency determine which wins.

## CI/CD for ML

CI/CD for ML automates testing and deployment of models. Unit tests validate data transforms; integration tests run the full pipeline.

## Containerisation

Containerisation with Docker packages model code and dependencies into a portable image. Kubernetes orchestrates container deployment at scale.

## Feature stores

Feature stores centralise feature definitions and computation so training and serving use identical features. Tecton and Feast are popular options.

## SHAP explanations

SHAP explains individual predictions by computing each feature contribution. It is used for model debugging, regulatory compliance, and stakeholder trust.

## Synthetic data

Synthetic data generation creates artificial datasets that preserve the statistical properties of real data. Useful for testing and privacy protection.

## Model compression

Model compression techniques like pruning, quantisation, and knowledge distillation shrink models for edge deployment with minimal accuracy loss.

## Streamlit and Hugging Face

Streamlit Community Cloud hosts Streamlit apps for free with 1GB RAM. Hugging Face Spaces offers more generous limits and free GPU tiers.
