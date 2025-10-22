# CodeMetaWizard

A proof-of-concept REACT-style LiteLLM agent that automatically extracts structured software metadata from code repositories using natural language reasoning and tool-based exploration.

## ⚠️ This project is entirely vibe-coded. It is not optimized for anything. It will cost you money to run it.
Read the code before you run it.

## Features

- 🤖 **Custom REACT Loop**: Implements reasoning and action cycles using LiteLLM's completion API
- 🔧 **Auto-Generated Tool Schemas**: Function schemas derived from docstrings via `@tool_schema` decorator
- 🔒 **Path Security**: Prevents directory traversal and absolute path access with built-in validation
- 📝 **Structured Outputs**: Uses Pydantic models for type-safe JSON schema validation
- 🌐 **GitHub Support**: Can fetch files directly from GitHub repositories
- 🎯 **Smart Extraction**: LLM decides which files to read and how to extract metadata

## Installation

```bash
# Clone the repository
git clone https://github.com/asuworks/codemetawizard.git
cd codemetawizard

# Install dependencies with uv
uv sync
```

## Usage

### Basic Usage

```bash
# Extract metadata from current directory
uv run python main.py --path .

# Specify output file
uv run python main.py --path . --output my_codemeta.json

# Adjust number of agent reasoning turns
uv run python main.py --path . --max-turns 15

# Want to see some hallucinations???
uv run python main.py --path . --max-turns 1
```

### Environment Variables

Create a `.env` file with your API credentials:

```bash
# OpenRouter (recommended)
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# Or use LiteLLM-Proxy endpoint
LITELLM_API_KEY=your_api_key_here
LITELLM_API_BASE=https://localhost:4000

# Optional: GitHub token for private repos
GITHUB_TOKEN=your_github_token
```

### Command Line Options

```bash
python main.py --help

Options:
  --path, -p PATH       Local directory path to scan (required)
  --max-turns INT       Maximum agent reasoning turns (default: 8)
  --model MODEL         LLM model name (default: openrouter/moonshotai/kimi-k2-0905)
  --api-key KEY         API key for LiteLLM (overrides env var)
  --output, -o FILE     Output JSON file (default: codemeta.llm.json)
```

## How It Works

1. **Initialization**: Agent receives system prompt with CodeMeta schema and available tools
2. **REACT Loop**: For each turn:
   - **Reason**: LLM analyzes current information and decides next action
   - **Act**: Calls tools to scan directories, read files, or fetch from GitHub
   - **Observe**: Receives tool results and incorporates into reasoning
3. **Extraction**: After exploration, LLM produces structured CodeMeta JSON
4. **Validation**: Pydantic validates output against CodeMeta schema

## Tools Available to Agent

- **`scan_tree_for_candidates`**: List files in directory (respects security boundaries)
- **`read_file`**: Read file contents (text only, with size limits)
- **`fetch_github_file`**: Fetch files from GitHub repositories

## Security Features

- ✅ Absolute path rejection
- ✅ Path traversal prevention
- ✅ Hidden file/folder exclusion
- ✅ Base directory containment checks
- ✅ Binary file detection

## Example Output

```json
{
  "CODEMETA": {
    "name": "codemetawizard",
    "version": "0.1.0",
    "description": "LLM-powered CodeMeta metadata extractor",
    "license": "MIT",
    "author": [
      {"name": "Humphrey Bogart"},
      {"name": "Cary Grant"}
    ],
    "contributor": [
      {"name": "Katharine Hepburn"},
      {"name": "Grace Kelly"}
    ],
    "repository": {
      "codeRepository": "https://github.com/asuworks/codemetawizard"
    }
  },
  "PROVENANCE": {
    "name": {"source": "pyproject.toml", "confidence": 1.0},
    "version": {"source": "pyproject.toml", "confidence": 1.0}
  }
}
```

## Architecture

### Custom REACT Implementation

Unlike traditional agent frameworks, CodeMetaWizard uses a custom REACT loop with:

- Direct LiteLLM completion API calls
- Explicit message history management
- Tool schema auto-generation from Python functions
- Pydantic-based structured outputs

### Tool Schema Decorator

The `@tool_schema` decorator automatically generates LiteLLM function calling schemas from:
- Function signatures with type hints
- Google-style docstring `Args:` sections
- Parameter defaults and optional types

```python
@tool_schema
def read_file(path: str, max_chars: Optional[int] = None) -> Dict[str, Any]:
    """
    Read and return raw file contents (text only).
    
    Args:
        path: File path to read (relative to base directory)
        max_chars: Optional maximum characters to read (default: 50000)
    """
    # Implementation...
```

## Contributing

Contributions welcome! Please ensure:
- All tests pass
- New tools include security checks
- Docstrings follow Google style
- Type hints are provided

## License

MIT License

## Authors

- Humphrey Bogart
- Cary Grant
- James Stewart

## Contributors

- Katharine Hepburn
- Grace Kelly
- Audrey Hepburn
- Marlene Dietrich
- Bette Davis

## Acknowledgments

- Built with [LiteLLM](https://github.com/BerriAI/litellm) for unified LLM API access
- Inspired by the [CodeMeta](https://codemeta.github.io/) standard
- Uses [Pydantic](https://docs.pydantic.dev/) for data validation
- Rich terminal output via [Rich](https://github.com/Textualize/rich)
