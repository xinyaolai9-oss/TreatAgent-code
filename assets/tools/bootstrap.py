#!/usr/bin/env python3
import json
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ClassificationBootstrapAnalyzer:
    def __init__(self, data_path: str, n_bootstrap: int = 1000, random_seed: int = 42):
        self.data_path = data_path
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.data = None
        self.true_labels = None
        self.predictions = None
        self.load_data()

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as handle:
            self.data = json.load(handle)
        self.true_labels = [item["label"] for item in self.data["results"]]
        self.predictions = [item["prediction_binary"] for item in self.data["results"]]

    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    def bootstrap_sample(self) -> Tuple[List[int], List[int]]:
        n_samples = len(self.true_labels)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = [self.true_labels[i] for i in indices]
        y_pred_boot = [self.predictions[i] for i in indices]
        return y_true_boot, y_pred_boot

    def calculate_bootstrap_std(self) -> Dict[str, Dict[str, float]]:
        print(f"Running bootstrap analysis with {self.n_bootstrap} samples...")
        bootstrap_results = {"accuracy": [], "f1": [], "precision": [], "recall": []}

        for i in range(self.n_bootstrap):
            if i % 100 == 0:
                print(f"Processing bootstrap sample {i}/{self.n_bootstrap}")

            y_true_boot, y_pred_boot = self.bootstrap_sample()
            metrics = self.calculate_metrics(y_true_boot, y_pred_boot)

            for metric in bootstrap_results:
                bootstrap_results[metric].append(metrics[metric])

        results = {}
        for metric in bootstrap_results:
            values = np.array(bootstrap_results[metric])
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            se_val = float(np.std(values) / np.sqrt(self.n_bootstrap))

            print(f"{metric:12} | bootstrap_mean: {mean_val:8.4f} | bootstrap_se: {se_val:8.4f}")

            results[metric] = {
                "bootstrap_mean": mean_val,
                "bootstrap_std": std_val,
                "standard_error": se_val,
                "ci_95_lower": float(np.percentile(values, 2.5)),
                "ci_95_upper": float(np.percentile(values, 97.5)),
                "relative_std": float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0.0,
            }

        return results

    def original_metrics(self) -> Dict[str, float]:
        return self.calculate_metrics(self.true_labels, self.predictions)

    def detailed_report(self) -> str:
        original = self.original_metrics()
        bootstrap = self.calculate_bootstrap_std()

        report = []
        report.append("=" * 60)
        report.append("CLASSIFICATION METRICS BOOTSTRAP ANALYSIS")
        report.append("=" * 60)
        report.append(f"Dataset: {self.data_path}")
        report.append(f"Total samples: {len(self.true_labels)}")
        report.append(f"Bootstrap iterations: {self.n_bootstrap}")
        report.append("")
        report.append("METRICS SUMMARY:")
        report.append("-" * 60)
        report.append(f"{'Metric':<12} {'Original':<10} {'Bootstrap':<10} {'Std Dev':<10} {'Std Error':<10} {'95% CI'}")
        report.append("-" * 60)

        for metric in ["accuracy", "precision", "recall", "f1"]:
            original_val = original[metric]
            boot_mean = bootstrap[metric]["bootstrap_mean"]
            boot_std = bootstrap[metric]["bootstrap_std"]
            boot_se = bootstrap[metric]["standard_error"]
            ci_lower = bootstrap[metric]["ci_95_lower"]
            ci_upper = bootstrap[metric]["ci_95_upper"]
            report.append(
                f"{metric:<12} {original_val:<10.4f} {boot_mean:<10.4f} {boot_std:<10.4f} "
                f"{boot_se:<10.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
            )

        return "\n".join(report)


if __name__ == "__main__":
    model = "gpt-4o"
    name_options = [
        "results_direct.json",
        "results_cot.json",
        "results_multiagent.json",
        "results_multiagent_merged.json",
    ]
    name = name_options[3]
    data_path = f"./results/{model}/{name}"

    analyzer = ClassificationBootstrapAnalyzer(data_path, n_bootstrap=1000)
    analyzer.original_metrics()
    analyzer.calculate_bootstrap_std()
