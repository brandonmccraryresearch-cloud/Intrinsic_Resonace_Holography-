# IO Directory - Runtime Logs and Failure Analysis

**Purpose**: Capture all computation failures and runtime logs for ML-based analysis and debugging.

## Directory Structure

```
io/
├── failures/          # Failed computation logs (JSON format)
│   ├── rg_flow_*.json       # RG flow integration failures
│   ├── observable_*.json    # Observable calculation failures
│   ├── topology_*.json      # Topological computation failures
│   └── general_*.json       # Other failures
├── logs/              # Structured runtime logs
└── README.md
```

## Failure Log Format

Each failure log is a JSON file containing:

```json
{
  "timestamp": "2025-12-27T04:30:00.000Z",
  "computation": "rg_flow_integration",
  "theoretical_ref": "IRH v21.4 Part 1 §1.2, Eq. 1.12",
  "error_type": "ConvergenceError",
  "error_message": "RG flow integration failed to converge",
  "parameters": {
    "initial_couplings": [50.0, 100.0, 150.0],
    "t_range": [-10, 10],
    "method": "RK45"
  },
  "stack_trace": "...",
  "context": {
    "session_id": "...",
    "notebook": "exascale_full_repo_ultra.ipynb",
    "cell_number": 15
  },
  "suggested_fixes": []
}
```

## Usage in Notebooks

```python
from src.logging.structured_logger import StructuredLogger
from datetime import datetime
import json

# Initialize logger
logger = StructuredLogger(output_dir="io/failures")

try:
    # Your computation
    result = compute_rg_flow(...)
except Exception as e:
    # Log failure
    logger.log_failure(
        computation="rg_flow_integration",
        error=e,
        parameters=params,
        theoretical_ref="IRH v21.4 Part 1 §1.2"
    )
    raise
```

## Gemini Integration (Colab)

Failed runs are automatically analyzed by Gemini (in Colab environment) to suggest fixes:

```python
# In Colab notebook cell
from google.colab import gemini
import json

# Read latest failure
with open("io/failures/latest.json") as f:
    failure = json.load(f)

# Ask Gemini for suggestions
prompt = f"""
Analyze this IRH computation failure and suggest fixes:

Computation: {failure['computation']}
Theoretical Reference: {failure['theoretical_ref']}
Error: {failure['error_message']}
Parameters: {json.dumps(failure['parameters'], indent=2)}

Suggest specific code refactoring to fix this issue.
"""

suggestions = gemini.generate(prompt)
print(suggestions)
```

## Automatic Push to Repository

All failure logs are automatically committed and pushed to the repository for collaborative debugging:

```bash
cd io/failures
git add *.json
git commit -m "Failure log: $(date +%Y%m%d_%H%M%S)"
git push origin main
```

This happens automatically in the ultra notebook when `auto_push_failures=True`.

---

**Last Updated**: December 2025  
**Part of**: IRH v21.4 Exascale Full Repository Ultra Notebook
