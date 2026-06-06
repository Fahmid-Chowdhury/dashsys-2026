# DASHSys 2026 Budget-Aware SQL/API Agent

This repository contains our submission system for the DASHSys 2026 workshop competition.

The system takes a natural-language query, selects the relevant SQL schema/API/example context, builds the filled agent prompt, runs a SQL/API agent, and saves the required deliverables for each query.

---

## Required Deliverables

For each query, the system generates the following required files:

| Organizer Requirement        | Generated File                   |
| ---------------------------- | -------------------------------- |
| Metadata JSON                | `metadata.json`                |
| Filled system prompt         | `filled_prompt.txt`            |
| Agent output trajectory JSON | `agent_output_trajectory.json` |

For single-query runs, these files are saved under:

```text
custom_query_runs/
```

Example output folder:

```text
custom_query_runs/
└── query_001_20260607_024321_list_all_journeys/
    ├── metadata.json
    ├── filled_prompt.txt
    ├── filled_prompt_messages.json
    ├── agent_output_trajectory.json
    ├── trace_only.json
    ├── answer.txt
    ├── summary.json
    ├── query.txt
    └── step_prompts/
```

The main files to check are:

```text
metadata.json
filled_prompt.txt
agent_output_trajectory.json
```

---

## Repository Structure

```text
dashsys-2026/
├── src/                    # Main source code
├── data/                   # Input files, indexes, and test outputs
├── DBSnapshot/             # Local parquet database snapshot
├── openapi_specs/          # OpenAPI specification files
├── custom_query_runs/      # Per-query deliverable folders
├── run_custom_query.py     # Runs one query and saves deliverables
└── README.md
```

---

## Main Source Files

```text
src/
├── metadata_generator.py   # Selects query-specific schema/API/example context
├── prompt_builder.py       # Builds the filled system prompt
├── agent_runner.py         # Runs the SQL/API action agent
├── answer_agent.py         # Writes final answer from verified evidence
├── tool_executor.py        # Executes SQL and API calls
├── verifier.py             # Validates SQL/API calls before execution
├── route_policy.py         # Controls route-aware tool usage
├── batch_runner.py         # Runs multiple test queries
├── schema_index.py         # Builds SQL schema index from DBSnapshot
├── example_index.py        # Builds example/retrieval index from data.json
├── hybrid_retriever.py     # Retrieves similar examples
└── openapi_indexer.py      # Builds API index from OpenAPI specs
```

---

## Data and Index Files

```text
data/
├── data.json                # Labeled examples provided by organizers
├── test.json                # Test queries
├── schema_index.json        # Generated SQL schema index
├── example_index.json       # Generated example index
├── api_index_enriched.json  # Generated API/OpenAPI index
├── retrieval_index/         # Retrieval files used for example search
├── test_metadata/           # Metadata files from batch runs
├── test_runs/               # Agent trajectory files from batch runs
└── test_runs_summary.json   # Batch run summary
```

---

## Setup

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install duckdb pandas numpy scikit-learn joblib python-dotenv requests pyyaml ollama google-genai sentence-transformers
```

---

## Environment Variables

Create a `.env` file in the project root.

For Ollama:

```env
LLM_PROVIDER=ollama
MODEL=gemma4:latest

ANSWER_LLM_PROVIDER=ollama
ANSWER_MODEL=gemma4:latest
```

For Gemini:

```env
LLM_PROVIDER=gemini
MODEL=gemini-3.5-flash

ANSWER_LLM_PROVIDER=gemini
ANSWER_MODEL=gemini-3.5-flash

GOOGLE_API_KEY=your_google_api_key
```

For real Adobe API execution, also add:

```env
ADOBE_CLIENT_ID=your_client_id
ADOBE_CLIENT_SECRET=your_client_secret
ADOBE_ORG_ID=your_org_id
ADOBE_SANDBOX_NAME=your_sandbox_name
ADOBE_SCOPES=openid,AdobeID,read_organizations,additional_info.projectedProductContext,session
ADOBE_IMS_TOKEN_URL=https://ims-na1.adobelogin.com/ims/token/v3
ADOBE_API_BASE_URL=https://platform.adobe.io
```

The `.env` file should not be committed.

---

## Build Required Indexes

Before running the agent, build the schema, example, and API indexes.

Run these commands from the project root:

```bash
python -m src.schema_index
python -m src.example_index
python -m src.openapi_indexer
```

These commands generate or update:

```text
data/schema_index.json
data/example_index.json
data/api_index_enriched.json
data/retrieval_index/
```

---

## Run One Query

To run a single query:

```bash
python run_custom_query.py --query "List all journeys"
```

The system will create a new timestamped folder under:

```text
custom_query_runs/
```

The folder will contain the required deliverables:

```text
metadata.json
filled_prompt.txt
agent_output_trajectory.json
```

It will also include supporting files such as:

```text
filled_prompt_messages.json
trace_only.json
answer.txt
summary.json
query.txt
step_prompts/
```

---

## Run All Test Queries with Batch Runner

The batch runner is used to run multiple test queries from a JSON file.

Default input file:

```text
data/test.json
```

Run all test queries:

```bash
python -m src.batch_runner
```

The batch runner will generate:

```text
data/test_metadata/
data/test_runs/
data/test_runs_summary.json
data/test_runs_metrics.json
data/test_runs_metrics.csv
```

For each query, batch mode saves:

```text
data/test_metadata/metadata_001.json
data/test_runs/run_001.json
```

The query number increases for each test query:

```text
metadata_001.json
metadata_002.json
metadata_003.json
...

run_001.json
run_002.json
run_003.json
...
```

The batch runner also saves prompt/debug information during agent execution under:

```text
data/agent_prompts/
```

---

## Batch Runner Options

Use real Adobe API calls instead of mocked API calls:

```bash
python -m src.batch_runner --real-api
```

***By default, batch mode uses mocked API calls. Real API execution requires valid Adobe credentials in `.env`.***

Run only the first 5 queries:

```bash
python -m src.batch_runner --limit 5
```

Skip the first 5 queries and run from the 6th query:

```bash
python -m src.batch_runner --start 5
```

Use a custom test file:

```bash
python -m src.batch_runner --input data/test.json
```

Use custom output folders:

```bash
python -m src.batch_runner \
  --output-dir data/test_runs \
  --metadata-output-dir data/test_metadata \
  --summary-path data/test_runs_summary.json
```

Use custom metrics output paths:

```bash
python -m src.batch_runner \
  --metrics-path data/test_runs_metrics.json \
  --metrics-csv-path data/test_runs_metrics.csv
```

Use a specific model/provider:

```bash
python -m src.batch_runner \
  --llm-provider ollama \
  --model gemma4:latest
```

Use Gemini:

```bash
python -m src.batch_runner \
  --llm-provider gemini \
  --model gemini-2.5-flash
```

Use separate models for action generation and final answer generation:

```bash
python -m src.batch_runner \
  --llm-provider ollama \
  --model gemma4:latest \
  --answer-llm-provider ollama \
  --answer-model gemma4:latest
```

Set agent limits:

```bash
python -m src.batch_runner \
  --max-steps 8 \
  --max-repair-attempts 3
```

---

## Batch Output Files

### `data/test_metadata/`

Contains one metadata file per query.

Example:

```text
data/test_metadata/metadata_001.json
```

This file contains the query-specific context selected by the system, including route, domain, intent, allowed SQL tables, allowed API endpoints, similar examples, and tool budget.

### `data/test_runs/`

Contains one full agent run file per query.

Example:

```text
data/test_runs/run_001.json
```

This file contains the agent result, including generated SQL/API actions, execution results, trace, debug events, final answer, and status.

### `data/test_runs_summary.json`

Contains one compact summary entry per query, including:

```text
- query id
- query text
- run status
- selected route
- selected domain and intent
- SQL call count
- API call count
- trace step count
- elapsed time
- answer preview
- failure reason, if any
```

### `data/test_runs_metrics.json`

Contains aggregate batch metrics, including:

```text
- total queries
- success count
- failure count
- failure rate
- success rate
- wall time
- average query time
- median query time
- average SQL calls per query
- average API calls per query
- average total tool calls per query
- budget satisfaction rate
```

### `data/test_runs_metrics.csv`

Contains the per-query metrics in CSV format for easier inspection and reporting.

---

## Organizer Checking Guide

To reproduce a single-query run from a fresh setup:

```bash
# 1. Install dependencies
pip install duckdb pandas numpy scikit-learn joblib python-dotenv requests pyyaml ollama google-genai sentence-transformers

# 2. Build indexes
python -m src.schema_index
python -m src.example_index
python -m src.openapi_indexer

# 3. Run a sample query
python run_custom_query.py --query "List all journeys"

# 4. Check generated files
custom_query_runs/<latest_query_folder>/
```

Inside the generated folder, the required deliverables are:

```text
metadata.json
filled_prompt.txt
agent_output_trajectory.json
```

To reproduce a batch run over the test set:

```bash
# 1. Build indexes
python -m src.schema_index
python -m src.example_index
python -m src.openapi_indexer

# 2. Run all test queries
python -m src.batch_runner	#By default, batch mode uses mocked API calls. 
or
python -m src.batch_runner --real-api	#Real API execution requires valid Adobe credentials in `.env`.

# 3. Check batch outputs
data/test_metadata/
data/test_runs/
data/test_runs_summary.json
data/test_runs_metrics.json
data/test_runs_metrics.csv
```

---

## What Each Required Deliverable Contains inside custom_query_runs/

### `metadata.json`

Contains the query-specific context selected by the system, including:

```text
- selected route
- selected domain and intent
- relevant SQL tables
- relevant API endpoints
- similar examples
- tool budget
```

### `filled_prompt.txt`

Contains the final populated prompt used by the action agent.

It includes:

```text
- user query
- selected metadata
- allowed SQL tables
- allowed API endpoints
- route-aware tool guidance
- output format instructions
```

### `agent_output_trajectory.json`

Contains the executed agent trajectory.

It records:

```text
- SQL calls generated by the agent
- API calls generated by the agent
- tool execution results
- verification status
- final answer
```

---

## Notes

- SQL queries are executed over local parquet files in `DBSnapshot/`.
- API specifications are loaded from `openapi_specs/`.
- Labeled examples are loaded from `data/data.json`.
- Test queries are loaded from `data/test.json`.
- The system supports SQL-only, API-only, SQL-plus-API, and multi-call SQL/API routes.
- The action agent generates SQL/API/final-answer actions.
- The answer agent writes the final answer only from verified SQL/API evidence.
- Single-query mode saves the exact three organizer deliverables in one folder.
- Batch mode is used to run the full test set and collect summaries/metrics.
- The system saves metadata, prompts, traces, answers, summaries, and metrics for reproducibility.
