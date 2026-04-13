# Data Science AI MCP Server
**By MEOK AI Labs** | [meok.ai](https://meok.ai)

ML/Data Science toolkit: feature importance ranking, model comparison, dataset profiling, correlation finding, and visualization recommendations.

## Tools

| Tool | Description |
|------|-------------|
| `feature_importance` | Rank features by estimated importance for ML tasks |
| `model_comparison` | Compare ML models across metrics with composite scoring |
| `dataset_profiler` | Profile dataset quality, completeness, and column statistics |
| `correlation_finder` | Pairwise Pearson correlations with multicollinearity warnings |
| `visualization_recommender` | Recommend charts based on data characteristics and goals |

## Installation

```bash
pip install mcp
```

## Usage

### Run the server

```bash
python server.py
```

### Claude Desktop config

```json
{
  "mcpServers": {
    "data-science": {
      "command": "python",
      "args": ["/path/to/data-science-ai-mcp/server.py"]
    }
  }
}
```

## Pricing

| Tier | Limit | Price |
|------|-------|-------|
| Free | 30 calls/day | $0 |
| Pro | Unlimited + premium features | $9/mo |
| Enterprise | Custom + SLA + support | Contact us |

## License

MIT
