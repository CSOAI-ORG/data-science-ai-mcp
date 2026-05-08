<div align="center">

# Data Science Ai MCP

**MCP server for data science ai mcp operations**

[![PyPI](https://img.shields.io/pypi/v/meok-data-science-ai-mcp)](https://pypi.org/project/meok-data-science-ai-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MEOK AI Labs](https://img.shields.io/badge/MEOK_AI_Labs-MCP_Server-purple)](https://meok.ai)

</div>

## Overview

Data Science Ai MCP provides AI-powered tools via the Model Context Protocol (MCP).

## Tools

| Tool | Description |
|------|-------------|
| `feature_importance` | Rank features by estimated importance for a prediction task. |
| `model_comparison` | Compare ML models across metrics. Returns composite scores, speed |
| `dataset_profiler` | Profile a dataset: completeness, quality issues, type distribution, |
| `correlation_finder` | Compute pairwise Pearson correlations between variables. Flags strong |
| `visualization_recommender` | Recommend visualizations based on data characteristics and analysis goal. |

## Installation

```bash
pip install meok-data-science-ai-mcp
```

## Usage with Claude Desktop

Add to your Claude Desktop MCP config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "data-science-ai-mcp": {
      "command": "python",
      "args": ["-m", "meok_data_science_ai_mcp.server"]
    }
  }
}
```

## Usage with FastMCP

```python
from mcp.server.fastmcp import FastMCP

# This server exposes 5 tool(s) via MCP
# See server.py for full implementation
```

## License

MIT © [MEOK AI Labs](https://meok.ai)
