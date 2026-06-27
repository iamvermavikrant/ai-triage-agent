# MCP Server Setup

## What is MCP?

The Model Context Protocol (MCP) is an open standard that allows AI models to call external tools in a structured, type-safe way. This project exposes two tools via an MCP server:

| Tool | Description |
|------|-------------|
| `fetch_test_logs` | Fetches CI test logs (GitHub Actions, local file, or mock) |
| `get_git_diff` | Retrieves unified git diff for a commit (GitHub API, local, or mock) |

## Running the MCP Server

### Stdio mode (default — for Claude Desktop / MCP clients)

```bash
python -m ai_triage_agent.mcp.server
```

### Configure in Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "ai-triage-agent": {
      "command": "python",
      "args": ["-m", "ai_triage_agent.mcp.server"],
      "cwd": "/path/to/ai-triage-agent",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "GITHUB_TOKEN": "ghp_...",
        "GITHUB_REPO": "owner/repo"
      }
    }
  }
}
```

## Tool Reference

### `fetch_test_logs`

**Description**: Fetches CI test logs for a given run ID.

**Input schema**:
```json
{
  "run_id": "string",       // GitHub Actions run ID, file path, or fixture key
  "backend": "string"       // "github_actions" | "local_file" | "mock"
}
```

**Backends**:

| Backend | Behavior |
|---------|----------|
| `github_actions` | Calls `GET /repos/{owner}/{repo}/actions/runs/{run_id}/logs`, extracts largest log file from ZIP |
| `local_file` | Reads `run_id` as a local file path |
| `mock` | Returns canned logs for fixture keys: `cuda_oom`, `import_error`, `regression_diff`, `flaky_timeout`, `env_mismatch` |

**Example**:
```json
{
  "run_id": "12345678",
  "backend": "github_actions"
}
```

### `get_git_diff`

**Description**: Retrieves the unified diff for a commit.

**Input schema**:
```json
{
  "commit_sha": "string",   // Full or short SHA
  "repo_path": "string",    // Optional: local git repo path
  "backend": "string"       // "github_api" | "local" | "mock"
}
```

**Backends**:

| Backend | Behavior |
|---------|----------|
| `github_api` | Calls `GET /repos/{owner}/{repo}/commits/{sha}` with `Accept: application/vnd.github.diff` |
| `local` | Runs `git diff {sha}^ {sha}` in `repo_path` |
| `mock` | Returns canned diffs for well-known fixture keys |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `GITHUB_TOKEN` | For real CI | GitHub PAT with `repo` scope |
| `GITHUB_REPO` | For real CI | `owner/repo` format |
| `MCP_HOST` | No | Server bind host (default: `127.0.0.1`) |
| `MCP_PORT` | No | Server port (default: `8765`) |

## Fallback Behavior

Both tools degrade gracefully:
- If `GITHUB_TOKEN` is not set, they log a warning and fall back to mock data.
- `get_git_diff` returns an empty string if no diff exists for a fixture (e.g. `flaky_timeout`), which causes `DiffAnalyzerNode` to return a low-risk placeholder rather than failing.
