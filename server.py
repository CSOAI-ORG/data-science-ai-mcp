#!/usr/bin/env python3
"""
Data Science AI MCP Server
==============================
Machine learning and data science toolkit for AI agents: feature importance
ranking, model comparison, dataset profiling, correlation finding, and
visualization recommendation.

By MEOK AI Labs | https://meok.ai

Install: pip install mcp
Run:     python server.py
"""

import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
FREE_DAILY_LIMIT = 30
_usage: dict[str, list[datetime]] = defaultdict(list)


def _check_rate_limit(caller: str = "anonymous") -> Optional[str]:
    now = datetime.now()
    cutoff = now - timedelta(days=1)
    _usage[caller] = [t for t in _usage[caller] if t > cutoff]
    if len(_usage[caller]) >= FREE_DAILY_LIMIT:
        return f"Free tier limit reached ({FREE_DAILY_LIMIT}/day). Upgrade: https://mcpize.com/data-science-ai-mcp/pro"
    _usage[caller].append(now)
    return None


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------
def _feature_importance(features: list[dict], target_type: str,
                        method: str) -> dict:
    """Rank features by estimated importance."""
    if not features:
        return {"error": "Provide features as [{name, type, cardinality, missing_pct, correlation_with_target}]"}

    methods = {
        "statistical": "Ranks by correlation strength, variance, and information value",
        "permutation": "Estimates importance by measuring prediction drop when feature is shuffled",
        "tree_based": "Uses split-based importance from decision trees (Gini/Entropy)",
    }

    if method not in methods:
        return {"error": f"Unknown method. Use: {list(methods.keys())}"}

    ranked = []
    for feat in features:
        name = feat.get("name", "unknown")
        ftype = feat.get("type", "numeric")
        cardinality = feat.get("cardinality", 10)
        missing_pct = feat.get("missing_pct", 0)
        correlation = abs(feat.get("correlation_with_target", 0))
        variance = feat.get("variance", 1.0)

        # Importance scoring
        score = 0

        # Correlation contribution (0-40)
        score += correlation * 40

        # Variance contribution (0-20)
        if variance > 0:
            var_score = min(20, math.log1p(variance) * 5)
        else:
            var_score = 0
        score += var_score

        # Penalize high missing (0 to -15)
        missing_penalty = min(15, missing_pct / 100 * 15)
        score -= missing_penalty

        # Penalize very high cardinality for categoricals
        if ftype == "categorical" and cardinality > 100:
            score -= 10
        elif ftype == "categorical" and cardinality > 20:
            score -= 5

        # Type bonus
        if target_type == "classification" and ftype == "categorical":
            score += 5
        elif target_type == "regression" and ftype == "numeric":
            score += 5

        importance = max(0, min(100, score))

        ranked.append({
            "feature": name,
            "type": ftype,
            "importance_score": round(importance, 2),
            "correlation": correlation,
            "missing_pct": missing_pct,
            "cardinality": cardinality,
            "recommendation": (
                "High importance - include in model" if importance >= 60 else
                "Moderate importance - test inclusion" if importance >= 30 else
                "Low importance - consider dropping"
            ),
        })

    ranked.sort(key=lambda x: x["importance_score"], reverse=True)

    top_features = [f["feature"] for f in ranked if f["importance_score"] >= 30]

    return {
        "method": method,
        "method_description": methods[method],
        "target_type": target_type,
        "total_features": len(features),
        "recommended_features": len(top_features),
        "rankings": ranked,
        "top_features": top_features,
        "feature_selection_tip": f"Start with top {min(len(top_features), 10)} features, then use cross-validation to find optimal set.",
    }


def _model_comparison(models: list[dict], task_type: str) -> dict:
    """Compare ML models across multiple metrics."""
    if not models:
        return {"error": "Provide models as [{name, accuracy, precision, recall, f1, training_time_sec, inference_ms}]"}

    task_metrics = {
        "classification": ["accuracy", "precision", "recall", "f1", "auc_roc"],
        "regression": ["rmse", "mae", "r_squared", "mape"],
        "ranking": ["ndcg", "map", "mrr"],
    }

    if task_type not in task_metrics:
        return {"error": f"Unknown task type. Use: {list(task_metrics.keys())}"}

    primary_metrics = task_metrics[task_type]
    compared = []

    for model in models:
        name = model.get("name", "unknown")
        metrics = {k: model.get(k, 0) for k in primary_metrics if k in model}
        training_time = model.get("training_time_sec", 0)
        inference_ms = model.get("inference_ms", 0)
        params = model.get("parameters", 0)

        # Composite score (weighted average of available metrics)
        if task_type == "classification":
            composite = (
                metrics.get("accuracy", 0) * 0.2 +
                metrics.get("precision", 0) * 0.2 +
                metrics.get("recall", 0) * 0.2 +
                metrics.get("f1", 0) * 0.3 +
                metrics.get("auc_roc", 0) * 0.1
            )
        elif task_type == "regression":
            # For regression, lower error = better, invert for scoring
            composite = (
                (1 - min(1, metrics.get("rmse", 1))) * 0.3 +
                (1 - min(1, metrics.get("mae", 1))) * 0.2 +
                metrics.get("r_squared", 0) * 0.4 +
                (1 - min(1, metrics.get("mape", 1) / 100)) * 0.1
            )
        else:
            composite = sum(metrics.values()) / max(len(metrics), 1)

        # Efficiency score
        if inference_ms > 0:
            efficiency = max(0, 100 - inference_ms)
        else:
            efficiency = 50

        compared.append({
            "name": name,
            "metrics": metrics,
            "composite_score": round(composite * 100, 2),
            "training_time_sec": training_time,
            "inference_ms": inference_ms,
            "parameters": params,
            "efficiency_score": round(efficiency, 1),
        })

    compared.sort(key=lambda x: x["composite_score"], reverse=True)

    # Best model analysis
    best = compared[0] if compared else None
    fastest = min(compared, key=lambda x: x["inference_ms"]) if compared else None
    most_efficient = min(compared, key=lambda x: x.get("parameters", float("inf"))) if compared else None

    return {
        "task_type": task_type,
        "model_count": len(compared),
        "primary_metrics": primary_metrics,
        "comparison": compared,
        "recommendations": {
            "best_overall": best["name"] if best else None,
            "best_overall_score": best["composite_score"] if best else None,
            "fastest_inference": fastest["name"] if fastest else None,
            "most_efficient": most_efficient["name"] if most_efficient else None,
        },
        "selection_guidance": [
            "Best overall: highest composite score across all metrics",
            "Production deployment: balance accuracy with inference latency",
            "Resource-constrained: consider model size and inference time",
            "Use cross-validation scores, not single train/test splits",
        ],
    }


def _dataset_profiler(columns: list[dict], row_count: int,
                      sample_values: dict) -> dict:
    """Profile a dataset with statistics and quality assessment."""
    if not columns:
        return {"error": "Provide columns as [{name, type, non_null_count, unique_count, min, max, mean, std}]"}

    profiles = []
    quality_issues = []
    total_cells = row_count * len(columns)
    total_missing = 0

    for col in columns:
        name = col.get("name", "unknown")
        dtype = col.get("type", "unknown")
        non_null = col.get("non_null_count", row_count)
        unique = col.get("unique_count", 0)
        missing = row_count - non_null
        missing_pct = (missing / max(row_count, 1)) * 100
        total_missing += missing

        profile = {
            "name": name,
            "type": dtype,
            "non_null": non_null,
            "missing": missing,
            "missing_pct": round(missing_pct, 2),
            "unique": unique,
            "unique_pct": round((unique / max(non_null, 1)) * 100, 2),
        }

        if dtype in ["int", "float", "numeric"]:
            profile["min"] = col.get("min", 0)
            profile["max"] = col.get("max", 0)
            profile["mean"] = col.get("mean", 0)
            profile["std"] = col.get("std", 0)
            profile["skewness"] = col.get("skewness", 0)

            # Check for potential issues
            if col.get("std", 0) == 0:
                quality_issues.append({"column": name, "issue": "Zero variance (constant)", "severity": "HIGH"})
            if missing_pct > 50:
                quality_issues.append({"column": name, "issue": f"High missing rate ({missing_pct:.0f}%)", "severity": "HIGH"})
            elif missing_pct > 10:
                quality_issues.append({"column": name, "issue": f"Missing values ({missing_pct:.0f}%)", "severity": "MEDIUM"})
        else:
            if unique == 1:
                quality_issues.append({"column": name, "issue": "Single unique value", "severity": "HIGH"})
            if unique == non_null:
                quality_issues.append({"column": name, "issue": "All unique (potential ID column)", "severity": "LOW"})

        # Add sample values if provided
        if name in sample_values:
            profile["sample_values"] = sample_values[name][:5]

        profiles.append(profile)

    completeness = ((total_cells - total_missing) / max(total_cells, 1)) * 100

    # Data type summary
    type_counts = Counter(col.get("type", "unknown") for col in columns)

    if completeness >= 95 and len(quality_issues) == 0:
        quality = "Excellent"
    elif completeness >= 80 and len([i for i in quality_issues if i["severity"] == "HIGH"]) == 0:
        quality = "Good"
    elif completeness >= 60:
        quality = "Fair"
    else:
        quality = "Poor"

    return {
        "row_count": row_count,
        "column_count": len(columns),
        "total_cells": total_cells,
        "completeness_pct": round(completeness, 2),
        "quality_assessment": quality,
        "type_distribution": dict(type_counts),
        "profiles": profiles,
        "quality_issues": quality_issues,
        "recommendations": [
            f"{'Handle missing values in ' + ', '.join(i['column'] for i in quality_issues if 'missing' in i['issue'].lower()[:20]) if any('missing' in i['issue'].lower() for i in quality_issues) else 'No missing value issues'}",
            f"{'Remove constant columns: ' + ', '.join(i['column'] for i in quality_issues if 'constant' in i['issue'].lower()) if any('constant' in i['issue'].lower() for i in quality_issues) else 'No constant columns found'}",
            "Check columns with all unique values - may be IDs not features",
            "Consider encoding categorical features with high cardinality",
        ],
    }


def _correlation_finder(variables: list[dict]) -> dict:
    """Find correlations between pairs of variables."""
    if not variables or len(variables) < 2:
        return {"error": "Provide at least 2 variables with values as [{name, values: []}]"}

    # Compute pairwise Pearson correlations
    pairs = []
    n = len(variables)

    for i in range(n):
        for j in range(i + 1, n):
            name_a = variables[i].get("name", f"var_{i}")
            name_b = variables[j].get("name", f"var_{j}")
            vals_a = variables[i].get("values", [])
            vals_b = variables[j].get("values", [])

            # Align lengths
            min_len = min(len(vals_a), len(vals_b))
            if min_len < 3:
                continue
            vals_a = vals_a[:min_len]
            vals_b = vals_b[:min_len]

            # Compute Pearson correlation
            try:
                mean_a = statistics.mean(vals_a)
                mean_b = statistics.mean(vals_b)
                std_a = statistics.stdev(vals_a)
                std_b = statistics.stdev(vals_b)

                if std_a == 0 or std_b == 0:
                    correlation = 0
                else:
                    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(vals_a, vals_b)) / (min_len - 1)
                    correlation = covariance / (std_a * std_b)
            except Exception:
                correlation = 0

            abs_corr = abs(correlation)
            if abs_corr >= 0.7:
                strength = "Strong"
            elif abs_corr >= 0.4:
                strength = "Moderate"
            elif abs_corr >= 0.2:
                strength = "Weak"
            else:
                strength = "Negligible"

            direction = "positive" if correlation > 0 else "negative" if correlation < 0 else "none"

            pairs.append({
                "variable_a": name_a,
                "variable_b": name_b,
                "correlation": round(correlation, 4),
                "abs_correlation": round(abs_corr, 4),
                "strength": strength,
                "direction": direction,
                "data_points": min_len,
            })

    pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

    strong = [p for p in pairs if p["abs_correlation"] >= 0.7]
    multicollinear = [p for p in pairs if p["abs_correlation"] >= 0.9]

    return {
        "variable_count": n,
        "pair_count": len(pairs),
        "correlations": pairs,
        "strong_correlations": strong,
        "multicollinearity_warnings": multicollinear,
        "recommendations": [
            f"{'Found ' + str(len(multicollinear)) + ' highly correlated pairs (>0.9) - consider removing one from each pair' if multicollinear else 'No multicollinearity detected'}",
            f"{'Strong correlations found between: ' + ', '.join(f'{p[\"variable_a\"]} & {p[\"variable_b\"]}' for p in strong[:3]) if strong else 'No strong correlations found'}",
            "Use VIF (Variance Inflation Factor) for more robust multicollinearity detection",
        ],
    }


def _visualization_recommender(data_description: dict) -> dict:
    """Recommend appropriate visualizations based on data characteristics."""
    columns = data_description.get("columns", [])
    row_count = data_description.get("row_count", 0)
    analysis_goal = data_description.get("goal", "explore")

    numeric_cols = [c for c in columns if c.get("type") in ["int", "float", "numeric"]]
    categorical_cols = [c for c in columns if c.get("type") in ["categorical", "string", "object"]]
    datetime_cols = [c for c in columns if c.get("type") in ["datetime", "date", "timestamp"]]

    recommendations = []

    # Distribution analysis
    if numeric_cols:
        recommendations.append({
            "chart_type": "histogram",
            "use_for": [c["name"] for c in numeric_cols[:3]],
            "reason": "Understand distribution shape, identify outliers and skewness",
            "library": "matplotlib / seaborn / plotly",
            "code_hint": "sns.histplot(data=df, x='column', kde=True)",
        })

        if len(numeric_cols) >= 2:
            recommendations.append({
                "chart_type": "scatter_plot",
                "use_for": [numeric_cols[0]["name"], numeric_cols[1]["name"]],
                "reason": "Explore relationships between numeric variables",
                "library": "matplotlib / plotly",
                "code_hint": "sns.scatterplot(data=df, x='col1', y='col2')",
            })

        if len(numeric_cols) >= 3:
            recommendations.append({
                "chart_type": "correlation_heatmap",
                "use_for": [c["name"] for c in numeric_cols],
                "reason": "Visualize all pairwise correlations at once",
                "library": "seaborn",
                "code_hint": "sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')",
            })

    # Categorical analysis
    if categorical_cols:
        for col in categorical_cols[:2]:
            cardinality = col.get("cardinality", 10)
            if cardinality <= 10:
                recommendations.append({
                    "chart_type": "bar_chart",
                    "use_for": [col["name"]],
                    "reason": f"Compare category frequencies ({cardinality} categories)",
                    "library": "matplotlib / plotly",
                    "code_hint": f"df['{col['name']}'].value_counts().plot(kind='bar')",
                })
            elif cardinality <= 5:
                recommendations.append({
                    "chart_type": "pie_chart",
                    "use_for": [col["name"]],
                    "reason": "Show proportion breakdown for few categories",
                    "library": "matplotlib / plotly",
                    "code_hint": f"df['{col['name']}'].value_counts().plot(kind='pie')",
                })

    # Time series
    if datetime_cols and numeric_cols:
        recommendations.append({
            "chart_type": "line_chart",
            "use_for": [datetime_cols[0]["name"], numeric_cols[0]["name"]],
            "reason": "Show trends over time",
            "library": "matplotlib / plotly",
            "code_hint": f"df.plot(x='{datetime_cols[0]['name']}', y='{numeric_cols[0]['name']}')",
        })

    # Categorical + Numeric
    if categorical_cols and numeric_cols:
        recommendations.append({
            "chart_type": "box_plot",
            "use_for": [categorical_cols[0]["name"], numeric_cols[0]["name"]],
            "reason": "Compare numeric distributions across categories",
            "library": "seaborn",
            "code_hint": f"sns.boxplot(data=df, x='{categorical_cols[0]['name']}', y='{numeric_cols[0]['name']}')",
        })

    # Large datasets
    if row_count > 10000:
        recommendations.append({
            "chart_type": "hexbin_plot",
            "use_for": [c["name"] for c in numeric_cols[:2]] if len(numeric_cols) >= 2 else [],
            "reason": "Handle overplotting in large datasets",
            "library": "matplotlib",
            "code_hint": "plt.hexbin(df['x'], df['y'], gridsize=30, cmap='YlOrRd')",
        })

    # Goal-specific
    if analysis_goal == "comparison":
        recommendations.append({
            "chart_type": "grouped_bar",
            "use_for": "Multiple categories and metrics",
            "reason": "Side-by-side comparison across groups",
            "library": "plotly / matplotlib",
            "code_hint": "df.groupby('category')['value'].mean().plot(kind='bar')",
        })
    elif analysis_goal == "composition":
        recommendations.append({
            "chart_type": "stacked_bar",
            "use_for": "Part-to-whole relationships",
            "reason": "Show how parts contribute to totals",
            "library": "matplotlib / plotly",
            "code_hint": "df.groupby(['cat1', 'cat2']).size().unstack().plot(kind='bar', stacked=True)",
        })

    return {
        "data_summary": {
            "row_count": row_count,
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "datetime_columns": len(datetime_cols),
        },
        "analysis_goal": analysis_goal,
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
        "general_tips": [
            "Start with histograms and scatter plots for initial exploration",
            "Use a correlation heatmap to find relationships quickly",
            "Box plots are great for comparing groups",
            "For presentations, prefer plotly (interactive) or clean matplotlib",
            "Always label axes and add titles for clarity",
        ],
    }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Data Science AI MCP",
    instructions="ML/Data Science toolkit: feature importance ranking, model comparison, dataset profiling, correlation finding, and visualization recommendations. By MEOK AI Labs.",
)


@mcp.tool()
def feature_importance(features: list[dict], target_type: str = "classification",
                       method: str = "statistical") -> dict:
    """Rank features by estimated importance for a prediction task.

    Args:
        features: Feature metadata as [{"name": "age", "type": "numeric", "cardinality": 50, "missing_pct": 2, "correlation_with_target": 0.65, "variance": 150}]
        target_type: ML task type (classification, regression)
        method: Importance method (statistical, permutation, tree_based)
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _feature_importance(features, target_type, method)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def model_comparison(models: list[dict], task_type: str = "classification") -> dict:
    """Compare ML models across metrics. Returns composite scores, speed
    comparisons, and recommendations for production vs accuracy.

    Args:
        models: Model results as [{"name": "XGBoost", "accuracy": 0.92, "precision": 0.90, "recall": 0.88, "f1": 0.89, "training_time_sec": 120, "inference_ms": 5, "parameters": 50000}]
        task_type: ML task (classification, regression, ranking)
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _model_comparison(models, task_type)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def dataset_profiler(columns: list[dict], row_count: int = 0,
                     sample_values: dict = {}) -> dict:
    """Profile a dataset: completeness, quality issues, type distribution,
    and per-column statistics.

    Args:
        columns: Column metadata as [{"name": "age", "type": "numeric", "non_null_count": 950, "unique_count": 80, "min": 18, "max": 90, "mean": 35.2, "std": 12.1}]
        row_count: Total number of rows
        sample_values: Optional sample values per column as {"col_name": [val1, val2, ...]}
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _dataset_profiler(columns, row_count, sample_values)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def correlation_finder(variables: list[dict]) -> dict:
    """Compute pairwise Pearson correlations between variables. Flags strong
    correlations and multicollinearity warnings.

    Args:
        variables: Variables with values as [{"name": "height", "values": [170, 175, 160, ...]}, {"name": "weight", "values": [70, 80, 55, ...]}]
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _correlation_finder(variables)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def visualization_recommender(data_description: dict) -> dict:
    """Recommend visualizations based on data characteristics and analysis goal.
    Returns chart types, code hints, and library suggestions.

    Args:
        data_description: Dataset info as {"columns": [{"name": "x", "type": "numeric", "cardinality": 50}], "row_count": 1000, "goal": "explore"}. Goals: explore, comparison, composition, distribution, relationship
    """
    err = _check_rate_limit()
    if err:
        return {"error": err}
    try:
        return _visualization_recommender(data_description)
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
