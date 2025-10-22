#!/usr/bin/env python3
"""
REACT-style Codemeta extractor CLI (LLM-driven parsing)

Key points:
- Tools only provide file listings and raw file contents (no local parsing heuristics).
- The LLM must decide which files to read, how to extract fields, and must return a single JSON object
  under the top-level key "CODEMETA" (Codemeta structure) plus "provenance" mapping for fields -> sources.
- Uses custom REACT loop with LiteLLM's completion API for function calling and structured outputs.
- Tool schemas are auto-generated from function signatures and docstrings via @tool_schema decorator.
- Validates final CODEMETA using Pydantic.

Usage:
    uv run main.py --path . --max-turns=15

Environment:
- OPENROUTER_API_KEY or LITELLM_API_KEY for LLM access
- OPENROUTER_API_BASE or LITELLM_API_BASE for custom API endpoints (optional)
- GITHUB_TOKEN for GitHub file fetches (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# LiteLLM for direct completion API
import litellm
import requests
from dotenv import load_dotenv
from litellm import completion
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# Enable JSON schema validation for structured outputs
litellm.enable_json_schema_validation = True

# pydantic for validation and structured outputs
from pydantic import BaseModel, Field, HttpUrl

# Load environment variables from .env file
load_dotenv()

# Initialize rich console
console = Console(force_terminal=True, force_interactive=False, legacy_windows=False)

# print entire messages before sending to LLM for debugging
DEBUG = False


def _print_messages(messages):
    if DEBUG:
        # Format messages for display
        messages_json = json.dumps(messages, indent=2)
        syntax = Syntax(
            messages_json,
            "json",
            theme="monokai",
            line_numbers=False,
            word_wrap=True,
        )

        console.print(
            Panel(
                syntax,
                title="[bold yellow]🔍 Messages to LLM[/bold yellow]",
                border_style="yellow",
                expand=False,
            )
        )


# -------------------------
# Pydantic Codemeta Model (validation only)
# -------------------------
class Person(BaseModel):
    name: str = Field(..., description="Full name of the person")
    affiliation: Optional[str] = Field(
        None, description="Institution or organization affiliation"
    )
    email: Optional[str] = Field(None, description="Email address")
    orcid: Optional[str] = Field(
        None, description="ORCID identifier (e.g., '0000-0002-1825-0097')"
    )


class Repository(BaseModel):
    codeRepository: Optional[HttpUrl] = Field(
        None, description="URL of the source code repository"
    )
    commit: Optional[str] = Field(
        None, description="Commit hash, tag, or branch reference"
    )


class Codemeta(BaseModel):
    name: Optional[str] = Field(None, description="Name of the software")
    version: Optional[str] = Field(None, description="Version number or identifier")
    description: Optional[str] = Field(
        None, description="Brief description of the software"
    )
    license: Optional[str] = Field(
        None, description="License identifier (e.g., 'MIT', 'Apache-2.0', 'GPL-3.0')"
    )
    author: Optional[List[Person]] = Field(None, description="List of software authors")
    contributor: Optional[List[Person]] = Field(
        None, description="List of software contributors"
    )
    keywords: Optional[List[str]] = Field(
        None, description="Keywords or tags describing the software"
    )
    softwareRequirements: Optional[Dict[str, Any]] = Field(
        None,
        description="Software requirements including programming language and dependencies",
    )
    repository: Optional[Repository] = Field(
        None, description="Source code repository information"
    )
    provenance: Optional[Dict[str, Any]] = Field(
        None, description="Provenance information mapping fields to their sources"
    )


class CodemetaOutput(BaseModel):
    """Wrapper model for structured output with CODEMETA and PROVENANCE."""

    CODEMETA: Codemeta = Field(..., description="Codemeta-compliant software metadata")
    PROVENANCE: Optional[Dict[str, Any]] = Field(
        None,
        description="Provenance information mapping each field to its source file and confidence",
    )


# -------------------------
# Agent system prompt (guiding LLM to produce structured Codemeta JSON)
# -------------------------
def get_agent_system_prompt() -> str:
    """Generate system prompt with Codemeta schema inferred from Pydantic model."""
    # Generate JSON schema from Pydantic model
    schema = Codemeta.model_json_schema()

    return f"""
You are a careful scientific software metadata extractor agent. Your job is to produce a Codemeta-style JSON object
describing the software in the provided directory or repository.

CODEMETA SCHEMA:
{json.dumps(schema, indent=2)}

RULES:
- You may ONLY use the tools available. Do not assume anything not supported by evidence from the files you read.
- Use `scan_tree_for_candidates` to list likely files. Call `read_file` to get raw text of any file. You may call
  `fetch_github_file` for public github content.
- When calling tools, use relative paths (e.g., ".", "./subdir", "filename.txt"). Absolute paths are discouraged.
- You are responsible for parsing the raw file text you receive and mapping it to Codemeta fields.
- Output MUST be a single JSON object and must include a top-level key "CODEMETA" with the codemeta object.
  Also include a top-level key "PROVENANCE" mapping each field you populated to the source file path / tool result and
  an optional confidence (0.0-1.0). Example:
  {{
    "CODEMETA": {{ "name": "...", "version": "...", "author": [ ... ] }},
    "PROVENANCE": {{ "name": {{"source":"path/to/README.md","confidence":0.9}}, ... }}
  }}
- Only populate fields where you have explicit evidence in the file contents you read.
- Prefer structured files (pyproject.toml, package.json, CITATION.cff, datacite.json) if present and use those as evidence.
- Do not output any extra explanation or notes outside the JSON object. The agent runner will capture your tool-calls and results.
- All fields in the schema above are optional unless marked as required. If you cannot find evidence for a field, omit it.
- IMPORTANT: On your FINAL turn, you MUST output the complete CODEMETA JSON object based on what you've learned, even if you haven't examined all files.

ACT in REACT style: Think briefly, call tools, observe results, think again, and produce the final JSON when confident.
"""


# -------------------------
# Tool schema decorator
# -------------------------
def tool_schema(func):
    """
    Decorator that extracts LiteLLM function schema from function signature and docstring.
    Parses Google-style docstring Args section for parameter descriptions.
    """
    import inspect
    from typing import get_args, get_origin

    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""

    # Extract description (first paragraph before Args:)
    description_lines = []
    for line in docstring.split("\n"):
        if line.strip().startswith("Args:"):
            break
        if line.strip():
            description_lines.append(line.strip())
    description = " ".join(description_lines)

    # Parse Args section for parameter descriptions
    param_docs = {}
    in_args = False
    current_param = None
    current_desc = []

    for line in docstring.split("\n"):
        line_stripped = line.strip()
        if line_stripped.startswith("Args:"):
            in_args = True
            continue
        if in_args:
            if line_stripped.startswith("Returns:") or (
                line_stripped and not line.startswith(" ")
            ):
                if current_param:
                    param_docs[current_param] = " ".join(current_desc)
                break
            if ":" in line_stripped and not line_stripped.startswith(" " * 8):
                # New parameter
                if current_param:
                    param_docs[current_param] = " ".join(current_desc)
                parts = line_stripped.split(":", 1)
                current_param = parts[0].strip()
                current_desc = [parts[1].strip()] if len(parts) > 1 else []
            elif current_param and line_stripped:
                # Continuation of description
                current_desc.append(line_stripped)

    if current_param:
        param_docs[current_param] = " ".join(current_desc)

    # Build properties from signature
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        param_type = param.annotation
        param_desc = param_docs.get(param_name, "")

        # Map Python types to JSON Schema types
        json_type = "string"
        items_type = None

        origin = get_origin(param_type)
        if origin is list or (
            isinstance(param_type, type) and issubclass(param_type, list)
        ):
            json_type = "array"
            args = get_args(param_type)
            if args:
                items_type = "string" if args[0] == str else "object"
        elif param_type == int or param_type == "int":
            json_type = "integer"
        elif param_type == bool or param_type == "bool":
            json_type = "boolean"
        elif origin is Optional or (
            hasattr(param_type, "__origin__") and param_type.__origin__ is type(None)
        ):
            # Optional parameter - extract inner type
            args = get_args(param_type)
            if args and args[0] != type(None):
                inner = args[0]
                if inner == int:
                    json_type = "integer"
                elif inner == bool:
                    json_type = "boolean"

        prop = {"type": json_type}
        if param_desc:
            prop["description"] = param_desc
        if json_type == "array" and items_type:
            prop["items"] = {"type": items_type}
        if param.default != inspect.Parameter.empty and param.default is not None:
            prop["default"] = param.default

        properties[param_name] = prop

        # Required if no default value and not Optional
        if param.default == inspect.Parameter.empty and origin is not Optional:
            required.append(param_name)

    # Store schema on function object
    func._tool_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    return func


# -------------------------
# Agent orchestration
# -------------------------
def create_tools(base_path: str):
    """
    Create tool functions with base_path baked in.
    This ensures all path arguments from the LLM are relative to the CLI-specified path.
    """

    base = Path(base_path).resolve()

    @tool_schema
    def scan_tree_for_candidates(
        path: str = ".", max_depth: int = 3, extensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Return candidate files (paths) under `path`. No content parsing — just file names and basic metadata.
        LLM will decide which files to call `read_file` on.

        Args:
            path: Directory path to scan (relative to base directory, default: ".")
            max_depth: Maximum directory depth to search (default: 3)
            extensions: Optional list of file extensions to filter (e.g., ['py', 'md', 'toml', 'json', 'yml', 'txt'])
                       If None, returns common metadata/config files
        """
        # Security: Resolve path relative to base and ensure it stays within base
        if path.startswith("/"):
            # Reject absolute paths - security risk
            console.print(f"[red]❌ Absolute paths not allowed:[/red] {path}")
            return {
                "error": "absolute_path_rejected",
                "path": path,
                "reason": "Security: only relative paths within base directory are allowed",
            }

        # Resolve path relative to base
        target = (base / path).resolve()

        # Security check: ensure resolved path is within base directory
        try:
            target.relative_to(base)
        except ValueError:
            # Path traversal attempt detected (e.g., ../../../etc/passwd)
            console.print(f"[red]❌ Path outside base directory:[/red] {target}")
            return {
                "error": "path_outside_base",
                "path": str(path),
                "reason": "Security: path must be within base directory",
            }

        console.print(
            f"[cyan]🔍 Scanning tree:[/cyan] {target.relative_to(base)} (max_depth={max_depth})"
        )

        if not target.exists():
            console.print(f"[red]❌ Path not found:[/red] {target.relative_to(base)}")
            return {"error": "path_not_found", "path": str(target.relative_to(base))}

        # Normalize extensions (remove dots, convert to lowercase)
        if extensions:
            exts = set()
            for e in extensions:
                e = e.lower().strip()
                # Remove leading dot if present
                e = e.lstrip(".")
                # If it looks like a full filename (contains dot), extract extension
                if "." in e:
                    e = e.split(".")[-1]
                exts.add(e)
        else:
            # Default: common metadata and config file extensions
            exts = None

        candidates = []
        base_depth = len(target.parts)
        for p in target.rglob("*"):
            # Security: Skip hidden files and folders (starting with .)
            # Check all path components to ensure no hidden directories in path
            if any(part.startswith(".") for part in p.parts):
                continue

            # Additional check: skip if filename itself starts with .
            if p.name.startswith("."):
                continue

            if p.is_file():
                depth = len(p.parts) - base_depth
                if depth > max_depth:
                    continue

                # Include file if:
                # - No extension filter specified (include common metadata files)
                # - File extension matches the filter
                # - File name matches common patterns (README, LICENSE, etc.)
                include_file = False

                p.name.lower()
                file_stem = p.stem.lower()
                file_ext = p.suffix.lstrip(".").lower()

                # Common metadata file patterns (always include these)
                metadata_patterns = [
                    "readme",
                    "license",
                    "citation",
                    "codemeta",
                    "metadata",
                    "package",
                    "setup",
                    "pyproject",
                    "cargo",
                    "gemfile",
                    "composer",
                    "pom",
                ]

                if exts is None:
                    # No filter - include common metadata/config files
                    if any(pattern in file_stem for pattern in metadata_patterns):
                        include_file = True
                    elif file_ext in [
                        "md",
                        "txt",
                        "json",
                        "toml",
                        "yaml",
                        "yml",
                        "cfg",
                        "ini",
                        "cff",
                    ]:
                        include_file = True
                else:
                    # With filter - check extension
                    if file_ext in exts:
                        include_file = True

                if include_file:
                    # Return path relative to base for consistency
                    rel_path = p.relative_to(base) if p.is_relative_to(base) else p
                    candidates.append(
                        {
                            "path": str(rel_path),
                            "size": p.stat().st_size,
                            "name": p.name,
                        }
                    )

        count = len(candidates)
        console.print(f"[green]✓ Found {count} candidate(s)[/green]")
        # console.print(f"[green]✓ Found {candidates} candidate(s)[/green]")
        return {"candidates": candidates, "count": count}

    @tool_schema
    def read_file(path: str, max_chars: Optional[int] = None) -> Dict[str, Any]:
        """
        Read and return raw file contents (text only). Does no parsing.
        LLM is responsible for interpreting the text.

        Args:
            path: File path to read (relative to base directory)
            max_chars: Optional maximum characters to read (default: 50000 if not specified)
        """
        if max_chars is None:
            max_chars = 50000

        # Security: Resolve path relative to base and ensure it stays within base
        if path.startswith("/"):
            # Reject absolute paths - security risk
            console.print(f"[red]❌ Absolute paths not allowed:[/red] {path}")
            return {
                "error": "absolute_path_rejected",
                "path": path,
                "reason": "Security: only relative paths within base directory are allowed",
            }

        # Resolve path relative to base
        target = (base / path).resolve()

        # Security check: ensure resolved path is within base directory
        try:
            target.relative_to(base)
        except ValueError:
            # Path traversal attempt detected
            console.print(f"[red]❌ Path outside base directory:[/red] {target}")
            return {
                "error": "path_outside_base",
                "path": str(path),
                "reason": "Security: path must be within base directory",
            }

        console.print(f"[cyan]📖 Reading file:[/cyan] {target.relative_to(base)}")

        if not target.exists():
            console.print(f"[red]❌ File not found:[/red] {target.relative_to(base)}")
            return {"error": "not_found", "path": str(target.relative_to(base))}
        try:
            # heuristics for binary detection
            with open(target, "rb") as fh:
                start = fh.read(4096)
            if b"\0" in start:
                console.print(f"[yellow]⚠️  Binary file detected:[/yellow] {target}")
                return {"path": str(target), "text": None, "note": "binary_file"}
            # read as text with ignore errors
            with open(target, "r", errors="ignore") as fh:
                text = fh.read(max_chars)
            truncated = len(text) >= max_chars
            console.print(
                f"[green]✓ Read {len(text)} chars[/green]"
                + (" [yellow](truncated)[/yellow]" if truncated else "")
            )
            return {"path": str(target), "text": text, "truncated": truncated}
        except Exception as e:
            return {"path": str(target), "error": str(e)}

    @tool_schema
    def fetch_github_file(
        raw_url_or_repo: str, path_within_repo: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        If user supplies a GitHub URL, this tool will attempt to fetch raw file contents from GitHub if possible.
        - If raw_url_or_repo is raw file URL (raw.githubusercontent.com), fetch it.
        - If it's a repo URL and path_within_repo provided, attempt to fetch raw content.
        This tool only returns raw file content; agent must interpret it.

        Args:
            raw_url_or_repo: GitHub URL (raw or repo)
            path_within_repo: Optional path within repo if using repo URL
        """
        console.print(f"[cyan]🌐 Fetching from GitHub:[/cyan] {raw_url_or_repo}")
        token = os.environ.get("GITHUB_TOKEN")
        headers = {"Accept": "application/vnd.github.v3.raw"}
        if token:
            headers["Authorization"] = f"token {token}"

        # If looks like raw.githubusercontent.com, fetch directly
        if "raw.githubusercontent.com" in raw_url_or_repo:
            try:
                r = requests.get(raw_url_or_repo, headers=headers, timeout=15)
                return {
                    "url": raw_url_or_repo,
                    "status": r.status_code,
                    "text": r.text if r.ok else None,
                }
            except Exception as e:
                return {"url": raw_url_or_repo, "error": str(e)}
        # If looks like a github repo url and path_within_repo is given, construct raw url
        if "github.com" in raw_url_or_repo and path_within_repo:
            # e.g. https://github.com/user/repo -> parse out user/repo
            # then build raw.githubusercontent.com/user/repo/main/{path_within_repo}
            parts = raw_url_or_repo.split("github.com/")[-1].strip("/").split("/")
            if len(parts) >= 2:
                user, repo = parts[0], parts[1]
                # guess branch is 'main' if not specified
                branch = "main"
                raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path_within_repo}"
                try:
                    r = requests.get(raw_url, headers=headers, timeout=15)
                    return {
                        "url": raw_url,
                        "status": r.status_code,
                        "text": r.text if r.ok else None,
                    }
                except Exception as e:
                    return {"url": raw_url, "error": str(e)}
        return {"error": "unsupported_input", "input": raw_url_or_repo}

    # Return tool functions and their auto-generated schemas
    tools = {
        "scan_tree_for_candidates": scan_tree_for_candidates,
        "read_file": read_file,
        "fetch_github_file": fetch_github_file,
    }

    # Extract schemas from decorated functions
    tool_schemas = [
        scan_tree_for_candidates._tool_schema,
        read_file._tool_schema,
        fetch_github_file._tool_schema,
    ]

    return tools, tool_schemas


def run_agent_with_retry(
    path: str,
    model: str,
    api_key: str,
    api_base: str | None,
    max_turns: int,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Wrapper that retries run_agent on failure with exponential backoff.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return run_agent(path, model, api_key, api_base, max_turns)
        except Exception as e:
            last_error = e
            error_str = str(e)
            console.print(
                f"\n[red]❌ Error on attempt {attempt + 1}/{max_retries}:[/red]"
            )
            console.print(f"[red]{error_str[:200]}[/red]\n")

            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                console.print(
                    f"[yellow]⏳ Retrying in {wait_time} seconds...[/yellow]\n"
                )
                time.sleep(wait_time)

    # All retries exhausted - try to generate output anyway
    console.print(f"[red]❌ All {max_retries} attempts failed.[/red]")
    console.print(
        "[yellow]⚠️  Attempting to generate CODEMETA from available information...[/yellow]"
    )

    # Return a minimal error structure with guidance
    return {
        "error": "All retries exhausted - tool execution failed",
        "CODEMETA": {
            "name": "Unknown",
            "description": "Failed to extract metadata due to API errors",
        },
        "PROVENANCE": {"error": str(last_error) if last_error else "Unknown error"},
    }


def run_agent(
    path: str, model: str, api_key: str, api_base: str | None, max_turns: int
) -> Dict[str, Any]:
    """
    Custom REACT loop using LiteLLM directly with function calling.
    """
    console.print(
        Panel.fit(
            f"[bold cyan]Starting Codemeta Extraction Agent[/bold cyan]\n"
            f"Model: [yellow]{model}[/yellow]\n"
            f"Path: [yellow]{path}[/yellow]\n"
            f"Max Turns: [yellow]{max_turns}[/yellow]\n"
            f"API Base: [yellow]{api_base or 'default'}[/yellow]",
            title="🤖 Agent Configuration",
            border_style="cyan",
        )
    )

    # Create tools with base path context
    tool_functions, tool_schemas = create_tools(path)

    # Initialize message history
    messages = [
        {"role": "system", "content": get_agent_system_prompt()},
        {
            "role": "user",
            "content": f"Please scan and extract Codemeta JSON from the directory: {path}",
        },
    ]

    console.print("\n[bold green]🚀 Starting REACT loop...[/bold green]\n")

    # REACT loop
    for turn in range(max_turns):
        console.print(f"[cyan]━━━ Turn {turn + 1}/{max_turns} ━━━[/cyan]")

        # print messages for debugging
        _print_messages(messages)

        try:
            # Call LiteLLM with function calling
            response = completion(
                model=model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                api_key=api_key,
                base_url=api_base,
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump())

            # Check if there are tool calls
            if (
                hasattr(assistant_message, "tool_calls")
                and assistant_message.tool_calls
            ):
                # Execute tool calls
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    console.print(
                        f"\n[yellow]� {tool_name}[/yellow]({json.dumps(tool_args, indent=2)[:100]}...)"
                    )

                    # Execute the tool
                    if tool_name in tool_functions:
                        try:
                            result = tool_functions[tool_name](**tool_args)
                            result_str = json.dumps(result)
                            tool_results.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": result_str,
                                }
                            )

                            # Display result preview
                            preview = (
                                result_str[:200] + "..."
                                if len(result_str) > 200
                                else result_str
                            )
                            console.print(f"[green]   → {preview}[/green]")

                        except Exception as e:
                            error_result = {"error": str(e)}
                            tool_results.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": json.dumps(error_result),
                                }
                            )
                            console.print(f"[red]   ❌ Error: {str(e)[:100]}[/red]")

                # Add tool results to messages
                messages.extend(tool_results)
                print("", flush=True)

            # Check if assistant provided content (thinking or final output)
            elif hasattr(assistant_message, "content") and assistant_message.content:
                content = assistant_message.content
                console.print(
                    f"\n[cyan]💬 Agent:[/cyan] [bright_black]{content[:150]}{'...' if len(content) > 150 else ''}[/bright_black]"
                )
                print("", flush=True)

                # Try to parse as CODEMETA JSON
                try:
                    # Look for JSON in the response
                    import re

                    json_match = re.search(r"\{[\s\S]*\}", content)
                    if json_match:
                        result = json.loads(json_match.group(0))
                        if "CODEMETA" in result or "name" in result:
                            console.print(
                                "\n[bold green]✅ Agent produced final output![/bold green]\n"
                            )
                            # Wrap if not already wrapped
                            if "CODEMETA" not in result:
                                result = {"CODEMETA": result}
                            return result
                except (json.JSONDecodeError, AttributeError):
                    pass

            # On last turn, force final output request
            if turn >= max_turns - 1:
                console.print(
                    "\n[yellow]⚠️  Max turns reached, requesting final output...[/yellow]"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "You've reached the maximum number of turns. Please output the complete CODEMETA JSON now based on what you've learned.",
                    }
                )

                # print messages for debugging
                _print_messages(messages)

                # One more call for final output with structured output (Pydantic model)
                try:
                    final_response = completion(
                        model=model,
                        messages=messages,
                        api_key=api_key,
                        base_url=api_base,
                        response_format=CodemetaOutput,
                    )

                    final_content = final_response.choices[0].message.content
                    if final_content:
                        # Parse and convert to dict
                        if isinstance(final_content, str):
                            result = json.loads(final_content)
                        else:
                            result = final_content
                        return result
                except Exception as e:
                    console.print(
                        f"[red]❌ Failed to get structured output: {str(e)[:100]}[/red]"
                    )
                    # Fallback to JSON mode
                    try:
                        final_response = completion(
                            model=model,
                            messages=messages,
                            api_key=api_key,
                            base_url=api_base,
                            response_format={"type": "json_object"},
                        )
                        final_content = final_response.choices[0].message.content
                        if final_content:
                            result = json.loads(final_content)
                            if "CODEMETA" not in result:
                                result = {"CODEMETA": result}
                            return result
                    except Exception as e2:
                        console.print(
                            f"[red]❌ Fallback also failed: {str(e2)[:100]}[/red]"
                        )
                        break

        except Exception as e:
            console.print(f"\n[red]❌ Error in turn {turn + 1}: {str(e)[:200]}[/red]\n")
            raise

    console.print("\n[bold green]✅ REACT loop completed![/bold green]\n")

    # If we get here without returning, create fallback output
    console.print(
        "[yellow]⚠️  No valid CODEMETA output generated, creating fallback...[/yellow]"
    )
    return {
        "CODEMETA": {
            "name": "unknown",
            "version": "0.0.0",
            "description": "Failed to extract metadata",
            "provenance": "extraction_failed",
        }
    }


# -------------------------
# CLI / main
# -------------------------
def validate_and_write(output_path: str, payload: Dict[str, Any]) -> None:
    """Validate CODEMETA with pydantic (best-effort) and write JSON."""
    cm = payload.get("CODEMETA")
    if not cm:
        # write what's available
        with open(output_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Wrote (no CODEMETA found) -> {output_path}")
        return
    # try validation (best-effort)
    try:
        validated = Codemeta(**cm)
        out = {"CODEMETA": validated.model_dump(mode="json", exclude_none=True)}
        # include provenance if present
        if "PROVENANCE" in payload:
            out["PROVENANCE"] = payload["PROVENANCE"]
        with open(output_path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"Wrote validated CODEMETA -> {output_path}")
    except Exception as e:
        # fallback: write the raw agent payload and a small note
        with open(output_path, "w") as fh:
            json.dump(
                {
                    "CODEMETA": cm,
                    "PROVENANCE": payload.get("PROVENANCE"),
                    "validation_error": str(e),
                },
                fh,
                indent=2,
            )
        print(f"Wrote CODEMETA with validation error (see file) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="REACT LLM-driven Codemeta extractor (standalone CLI)"
    )
    parser.add_argument(
        "--path", "-p", required=True, help="Local directory path or GitHub URL to scan"
    )
    parser.add_argument(
        "--max-turns", type=int, default=8, help="Max agent turns / iterations"
    )
    parser.add_argument(
        "--model",
        default="openrouter/moonshotai/kimi-k2-0905",
        help="LLM model name (when using proxy, don't include 'litellm/' prefix)",
    )
    parser.add_argument(
        "--api-key", type=str, required=False, help="API key for LiteLLM"
    )
    parser.add_argument(
        "--output", "-o", default="codemeta.llm.json", help="Output JSON file"
    )
    args = parser.parse_args()

    # Get API key from args, env var, or prompt
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        api_key = input("Enter an API key for LiteLLM: ")

    # Get API base URL from env var if available
    api_base = os.getenv("OPENROUTER_API_BASE")

    # Run the agent with retry logic (3 attempts by default)
    payload = run_agent_with_retry(
        args.path,
        model=args.model,
        api_key=api_key,
        api_base=api_base,
        max_turns=args.max_turns,
        max_retries=3,
    )
    validate_and_write(args.output, payload)


if __name__ == "__main__":
    main()
