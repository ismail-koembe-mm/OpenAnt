# Gemini 2.5 Flash Integration Guide

OpenAnt now supports **Google Gemini 2.5 Flash** (and other Google models) via Vertex AI or Google Generative AI. This guide explains how to set up and use Gemini with OpenAnt.

## Prerequisites

1. **Python 3.11+** (same as OpenAnt)
2. **Google Cloud Project** with Vertex AI or Generative AI API enabled
3. **Google Cloud credentials** (service account or API key)

## Setup Options

### Option 1: Vertex AI (Recommended for Production)

Best for enterprise use with full GCP integration.

#### Step 1: Set up Google Cloud Authentication

```bash
# Option A: Using service account (recommended)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
export GOOGLE_CLOUD_PROJECT=your-project-id

# Option B: Using gcloud CLI (local development)
gcloud auth application-default login
gcloud config set project your-project-id
```

#### Step 2: Enable Vertex AI API

```bash
gcloud services enable aiplatform.googleapis.com
```

#### Step 3: Configure OpenAnt for Vertex AI

```bash
openant set-provider google
openant set-google-project your-project-id
```

#### Step 4: Run with Gemini 2.5 Flash

```bash
openant analyze --llm-provider google --model gemini-2.5-flash
```

### Option 2: Google Generative AI (Simpler Setup)

Good for quick testing or if you have a direct API key.

#### Step 1: Get an API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy your API key

#### Step 2: Configure OpenAnt

```bash
export GOOGLE_API_KEY=your-api-key
openant set-provider google
```

#### Step 3: Run with Gemini

```bash
openant analyze --llm-provider google --model gemini-2.5-flash
```

## Configuration

### Setting the Default Provider

```bash
# Set Anthropic as default (original)
openant set-provider anthropic

# Set Google as default
openant set-provider google
```

### Per-Command Provider Override

```bash
# Use Google for this scan only
openant scan --llm-provider google --model gemini-2.5-flash

# Use Anthropic for this scan only
openant scan --llm-provider anthropic --model claude-opus-4-20250514
```

### Programmatic Use

```python
from utilities.llm_factory import create_llm_client

# Vertex AI
client = create_llm_client(
    provider="google",
    model="gemini-2.5-flash",
    project_id="your-project-id"
)

# Google Generative AI
client = create_llm_client(
    provider="google",
    model="gemini-2.5-flash",
    use_genai=True,
    api_key="your-api-key"
)

# Anthropic (original)
client = create_llm_client(
    provider="anthropic",
    model="claude-opus-4-20250514"
)
```

## Supported Models

### Google Vertex AI

- `gemini-2.5-flash` (recommended) - Fast, cost-effective
- `gemini-1.5-pro` - Higher quality, higher cost
- `gemini-1.5-flash` - Balanced

### Anthropic

- `claude-opus-4-20250514` (default) - Highest quality
- `claude-sonnet-4-20250514` - Cost-effective alternative

## Cost Comparison

### Gemini 2.5 Flash (Google)

- Input: $0.075 per million tokens
- Output: $0.30 per million tokens
- **~4x cheaper than Claude Opus**

### Claude Opus 4 (Anthropic)

- Input: $15.00 per million tokens
- Output: $75.00 per million tokens

## CLI Flags

### analyze, enhance, verify, scan commands

```bash
--llm-provider {anthropic,google}
  Specify LLM provider (default: anthropic)

--model MODEL
  Specify model name (default: claude-opus-4-20250514 for anthropic,
  gemini-2.5-flash for google)

--llm-config JSON
  Additional provider-specific config (e.g., for Google: project_id, location)
  Format: '{"project_id": "my-project", "location": "us-central1"}'
```

## Examples

### Full Pipeline with Gemini 2.5 Flash

```bash
# Initialize
openant init https://github.com/example/repo -l python --name example/repo

# Run full pipeline
openant scan --llm-provider google --model gemini-2.5-flash --verify

# View results
cat ~/.openant/projects/example/repo/scans/*/python/results_verified.json
```

### Mixed Provider Usage

```bash
# Use Gemini for detection (cheaper)
openant analyze --llm-provider google --model gemini-2.5-flash

# Use Claude for verification (better tool use)
openant verify --llm-provider anthropic

# Generate report
openant report
```

## Troubleshooting

### `GOOGLE_APPLICATION_CREDENTIALS not found`

**Vertex AI only.** Set the path to your service account JSON file:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### `GOOGLE_API_KEY not found`

**Google Generative AI only.** Set your API key:

```bash
export GOOGLE_API_KEY=your-api-key
```

### `google-cloud-aiplatform not installed`

Install Google dependencies:

```bash
pip install google-cloud-aiplatform>=1.45.0
```

### `google-generativeai not installed`

Install Google Generative AI SDK:

```bash
pip install google-generativeai>=0.6.0
```

### Model Not Found / 404 Error

Verify:
1. Model name is correct (case-sensitive)
2. Project ID is correct
3. Region supports the model (default: `us-central1`)
4. Vertex AI API is enabled

### Rate Limit (429 Error)

OpenAnt automatically backs off and retries. To adjust backoff time:

```bash
openant analyze --backoff-seconds 60
```

## Environment Variables

### Anthropic

- `ANTHROPIC_API_KEY`: Your Anthropic API key

### Google Vertex AI

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account JSON
- `GOOGLE_CLOUD_PROJECT`: GCP project ID
- `GOOGLE_CLOUD_LOCATION`: GCP region (default: `us-central1`)

### Google Generative AI

- `GOOGLE_API_KEY`: Your Google API key

## Important Notes

### Stage 2 Verification

Stage 2 (verification) currently requires Claude for tool use capabilities. When using Google for Stage 1 analysis:

```bash
# Stage 1: Use Gemini (faster, cheaper)
openant analyze --llm-provider google --model gemini-2.5-flash

# Stage 2: Uses Claude internally (tool calling)
openant verify
```

This hybrid approach optimizes cost (cheap detection + robust verification).

### Token Tracking

Token usage is tracked separately for each provider. View costs with:

```bash
openant report  # Shows total cost by model
```

### Prompt Compatibility

Prompts are optimized for both providers, but results may vary:

- **Gemini** tends to be more concise
- **Claude** tends to be more detailed

Consider testing both on your codebase to choose what works best.

## Updating to Newer Models

When new models become available, update the client:

```python
from utilities.gemini_client import MODEL_PRICING

# Add new model pricing
MODEL_PRICING["gemini-3.0-flash"] = {"input": 0.05, "output": 0.20}
```

Or use the CLI:

```bash
openant analyze --model gemini-3.0-flash
```

## Support

For issues specific to Gemini or Google Cloud:

1. Check [Google Generative AI documentation](https://ai.google.dev/)
2. Check [Vertex AI documentation](https://cloud.google.com/docs/generative-ai)
3. Review OpenAnt logs in `~/.openant/projects/*/scans/*/python/`
