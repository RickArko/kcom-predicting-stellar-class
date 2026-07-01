from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from stellar.blending import (
    align_proba,
    read_probability_artifact,
    score_proba,
    write_probability_artifact,
)
from stellar.foundation import build_feature_matrices, stratified_context_indices

ROOT = Path(__file__).resolve().parents[1]


def _load_script_func(script_name: str, func_name: str):
    spec = importlib.util.spec_from_file_location(script_name, ROOT / "scripts" / script_name)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


run_benchmark = _load_script_func("benchmark_tfm.py", "run_benchmark")
load_tfm_env = _load_script_func("benchmark_tfm.py", "_load_env")
run_blend = _load_script_func("blend_predictions.py", "run_blend")
collect_scores = _load_script_func("score_tfm.py", "collect_scores")
select_best = _load_script_func("select_best_submission.py", "select_best")


def _write_synthetic_csvs(data_dir):
    labels = np.array(["GALAXY", "STAR", "QSO"] * 30)
    offsets = {"GALAXY": 0.0, "STAR": 3.0, "QSO": 6.0}
    base = np.array([offsets[label] for label in labels])

    train = pd.DataFrame(
        {
            "id": np.arange(len(labels)),
            "u": 15.0 + base,
            "g": 14.0 + base * 0.8,
            "r": 13.0 + base * 0.6,
            "i": 12.0 + base * 0.4,
            "z": 11.0 + base * 0.2,
            "redshift": 0.1 + base * 0.05,
            "spectral_type": np.where(labels == "STAR", "G/K", "A/F"),
            "galaxy_population": np.where(labels == "GALAXY", "Red_Sequence", "Blue_Cloud"),
            "class": labels,
        }
    )
    test = train.drop(columns=["class"]).head(12).copy()
    test["id"] = np.arange(1000, 1012)

    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    return train, test


def test_align_proba_reorders_columns():
    proba = np.array([[0.2, 0.7, 0.1]])
    aligned = align_proba(proba, ["QSO", "GALAXY", "STAR"], ["GALAXY", "STAR", "QSO"])
    np.testing.assert_array_equal(aligned, np.array([[0.7, 0.1, 0.2]]))


def test_load_env_maps_tabpfn_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TABPFN_API_KEY", raising=False)
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    (tmp_path / ".env").write_text("TABPFN_API_KEY=test-key\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")

    load_tfm_env(str(config_path))

    assert os.environ["TABPFN_API_KEY"] == "test-key"
    assert os.environ["TABPFN_TOKEN"] == "test-key"
    assert os.environ["HF_TOKEN"] == "test-key"
    assert os.environ["HUGGINGFACE_HUB_TOKEN"] == "test-key"


def test_stratified_context_indices_keeps_classes():
    y = pd.Series(["GALAXY"] * 20 + ["STAR"] * 10 + ["QSO"] * 5)
    idx = stratified_context_indices(y, 9, random_state=42)
    sampled = set(y.iloc[idx])
    assert len(idx) == 9
    assert sampled == {"GALAXY", "STAR", "QSO"}


def test_build_domain_features_preserves_categoricals():
    train = pd.DataFrame(
        {
            "id": [1, 2],
            "u": [20.0, 21.0],
            "g": [19.0, 20.0],
            "redshift": [0.1, 0.2],
            "spectral_type": ["A/F", "G/K"],
            "class": ["GALAXY", "STAR"],
        }
    )
    test = train.drop(columns=["class"]).copy()
    X_train, X_test = build_feature_matrices(
        train,
        test,
        train["class"],
        {
            "mode": "domain",
            "target_col": "class",
            "drop_cols": ["id"],
            "color_pairs": [["u", "g"]],
            "interaction_pairs": [["redshift", "u_g"]],
            "cat_cols": ["spectral_type"],
            "encoding": "passthrough",
        },
    )
    assert "id" not in X_train.columns
    assert "u_g" in X_train.columns
    assert "redshift_x_u_g" in X_train.columns
    assert X_train["spectral_type"].tolist() == ["A/F", "G/K"]
    assert X_train.columns.tolist() == X_test.columns.tolist()


def test_probability_artifact_roundtrip(tmp_path):
    classes = ["GALAXY", "STAR", "QSO"]
    y_true = np.array(["GALAXY", "STAR", "QSO"])
    oof = np.eye(3)
    test = np.eye(3)[:2]
    run_dir = tmp_path / "artifact"

    write_probability_artifact(
        run_dir,
        config={"backend": {"name": "dummy"}},
        metrics={"overall_oof_score": 1.0},
        oof_proba=oof,
        test_proba=test,
        train_ids=np.arange(3),
        test_ids=np.array([10, 11]),
        classes=classes,
        y_true=y_true,
        predictions=np.array(["GALAXY", "STAR"]),
    )

    artifact = read_probability_artifact(run_dir)
    assert artifact.classes == classes
    assert artifact.metrics["overall_oof_score"] == 1.0
    np.testing.assert_array_equal(artifact.oof_proba, oof)
    np.testing.assert_array_equal(artifact.y_true, y_true)
    assert (run_dir / "submission.csv").exists()


def test_blend_predictions_selects_valid_artifact(tmp_path):
    classes = ["GALAXY", "STAR", "QSO"]
    y_true = np.array(["GALAXY", "STAR", "QSO", "GALAXY", "STAR", "QSO"])
    good = np.eye(3)[[0, 1, 2, 0, 1, 2]] * 0.9 + 0.05
    weak = np.full_like(good, 1.0 / 3.0)
    test_proba = np.eye(3)[:3]

    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    write_probability_artifact(
        run_a,
        config={"name": "a"},
        metrics={"overall_oof_score": score_proba(good, y_true, classes)},
        oof_proba=good,
        test_proba=test_proba,
        train_ids=np.arange(len(y_true)),
        test_ids=np.array([100, 101, 102]),
        classes=classes,
        y_true=y_true,
    )
    write_probability_artifact(
        run_b,
        config={"name": "b"},
        metrics={"overall_oof_score": score_proba(weak, y_true, classes)},
        oof_proba=weak,
        test_proba=weak[:3],
        train_ids=np.arange(len(y_true)),
        test_ids=np.array([100, 101, 102]),
        classes=classes,
        y_true=y_true,
    )

    blend_dir = run_blend(
        [str(run_a), str(run_b)],
        run_name="unit_blend",
        output_dir=str(tmp_path / "out"),
        per_class=True,
        tune_thresholds=True,
    )
    artifact = read_probability_artifact(blend_dir)
    assert artifact.metrics["overall_oof_score"] >= 0.99
    assert (blend_dir / "submission.csv").exists()


def test_score_tfm_collects_ranked_artifacts(tmp_path):
    classes = ["GALAXY", "STAR", "QSO"]
    y_true = np.array(["GALAXY", "STAR", "QSO"])
    proba = np.eye(3)

    for name, score in [("low", 0.2), ("high", 0.9)]:
        write_probability_artifact(
            tmp_path / name,
            config={"name": name},
            metrics={
                "overall_oof_score": score,
                "model_family": "dummy",
                "wall_time_seconds": 1.0,
            },
            oof_proba=proba,
            test_proba=proba,
            train_ids=np.arange(3),
            test_ids=np.arange(3),
            classes=classes,
            y_true=y_true,
        )

    df = collect_scores(tmp_path)
    assert df.iloc[0]["run"] == "high"
    assert df.iloc[0]["overall_oof_score"] == 0.9


def test_select_best_submission_across_runs_and_tfm(tmp_path):
    runs_dir = tmp_path / "runs"
    tfm_dir = tmp_path / "tfm"
    run_dir = runs_dir / "20260630_000001_tree"
    artifact_dir = tfm_dir / "20260630_000002_blend"
    run_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)

    with open(run_dir / "metrics.json", "w") as f:
        f.write('{"metrics": {"overall_oof_score": 0.8}}')
    pd.DataFrame({"id": [1], "class": ["GALAXY"]}).to_csv(run_dir / "submission.csv", index=False)

    write_probability_artifact(
        artifact_dir,
        config={"name": "blend"},
        metrics={"overall_oof_score": 0.9, "model_family": "blend"},
        oof_proba=np.eye(3),
        test_proba=np.eye(3)[:1],
        train_ids=np.arange(3),
        test_ids=np.array([1]),
        classes=["GALAXY", "STAR", "QSO"],
        y_true=np.array(["GALAXY", "STAR", "QSO"]),
        predictions=np.array(["GALAXY"]),
    )

    best = select_best(runs_dir, tfm_dir)
    assert best["source"] == "tfm"
    assert best["run"] == "20260630_000002_blend"
    assert best["score"] == 0.9
    assert best["path"].endswith("submission.csv")


def test_dummy_benchmark_runner_writes_artifact(tmp_path):
    data_dir = tmp_path / "data"
    _write_synthetic_csvs(data_dir)
    cfg = {
        "competition": {
            "target": "class",
            "classes": ["GALAXY", "STAR", "QSO"],
        },
        "paths": {
            "data": str(data_dir),
            "tfm_outputs": str(tmp_path / "tfm"),
        },
        "data": {},
        "features": {
            "mode": "engineered",
            "drop_cols": ["id"],
            "color_pairs": [["u", "g"], ["g", "r"], ["r", "i"], ["i", "z"]],
            "cat_cols": ["spectral_type", "galaxy_population"],
            "encoding": "ohe",
        },
        "cv": {
            "n_splits": 3,
            "shuffle": True,
            "random_state": 42,
        },
        "backend": {
            "name": "dummy",
            "params": {"max_iter": 1000, "random_state": 42},
        },
        "benchmark": {"tune_thresholds": True},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    run_dir = run_benchmark(str(config_path), run_name="dummy_unit")
    artifact = read_probability_artifact(run_dir)
    assert artifact.oof_proba.shape == (90, 3)
    assert artifact.test_proba.shape == (12, 3)
    assert artifact.metrics["model_family"] == "dummy"
    assert artifact.metrics["overall_oof_score"] > 0.9
    assert (run_dir / "submission.csv").exists()
