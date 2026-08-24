"""Generated deterministic interface for random_forest."""

from agentforge.generated_runtime import evaluate_candidate_model, predict_candidate, train_candidate

ALGORITHM = "random_forest"
HYPERPARAMETERS = {'n_estimators': 240, 'max_depth': 10, 'min_samples_leaf': 3, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample', 'random_state': 42, 'n_jobs': 1}


def train(data_path: str, model_path: str) -> dict:
    return train_candidate(ALGORITHM, data_path, model_path)


def predict(model_path: str, data_path: str) -> list:
    return predict_candidate(model_path, data_path)


def evaluate(model_path: str, data_path: str) -> dict:
    return evaluate_candidate_model(model_path, data_path)
