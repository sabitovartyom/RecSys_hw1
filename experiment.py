"""Небольшие циклы подбора на validation и финального сравнения на test."""

import gc
import json
from time import perf_counter

import pandas as pd

from evaluation import evaluate, prepare_evaluation


def fit_and_evaluate(model_class, params, train, shape, item_features, evaluation):
    start = perf_counter()
    model = model_class(**params).fit(train, shape, item_features)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    metrics, per_user = evaluate(model, *evaluation)
    metrics["fit_seconds"] = fit_seconds
    metrics["eval_seconds"] = perf_counter() - start
    return model, metrics, per_user


def tune_models(specs, train, valid, shape, item_features):
    """specs: {имя: (класс модели, список словарей параметров)}."""
    evaluation = prepare_evaluation(train, valid, shape)
    results, best_params = [], {}
    for name, (model_class, configs) in specs.items():
        best_score = -1
        for params in configs:
            print(f"{name}: {params}", flush=True)
            model, metrics, _ = fit_and_evaluate(
                model_class, params, train, shape, item_features, evaluation
            )
            results.append({"model": name, "params": json.dumps(params), **metrics})
            if metrics["NDCG@10"] > best_score:
                best_score = metrics["NDCG@10"]
                best_params[name] = params.copy()
            del model
            gc.collect()
    return pd.DataFrame(results), best_params


def final_comparison(specs, best_params, train_valid, test, shape, item_features):
    evaluation = prepare_evaluation(train_valid, test, shape)
    results, user_results = [], {}
    for name, (model_class, _) in specs.items():
        print(f"Финальное обучение: {name}", flush=True)
        model, metrics, per_user = fit_and_evaluate(
            model_class, best_params[name], train_valid, shape, item_features, evaluation
        )
        results.append({"model": name, "params": json.dumps(best_params[name]), **metrics})
        user_results[name] = per_user
        del model
        gc.collect()
    table = pd.DataFrame(results).sort_values("NDCG@10", ascending=False)
    return table, user_results
