"""Generated deterministic interface for logistic_regression."""

from agentforge.generated_runtime import evaluate_candidate_model, predict_candidate, train_candidate

ALGORITHM = "logistic_regression"
HYPERPARAMETERS = {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42}


def train(data_path: str, model_path: str) -> dict:
    return train_candidate(ALGORITHM, data_path, model_path)


def predict(model_path: str, data_path: str) -> list:
    return predict_candidate(model_path, data_path)


def evaluate(model_path: str, data_path: str) -> dict:
    return evaluate_candidate_model(model_path, data_path)
