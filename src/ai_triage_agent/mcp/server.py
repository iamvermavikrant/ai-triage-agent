"""MCP server exposing fetch_test_logs and get_git_diff as tools."""

from __future__ import annotations

import os

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

from ai_triage_agent.mcp.tools.fetch_test_logs import fetch_test_logs
from ai_triage_agent.mcp.tools.get_git_diff import get_git_diff

log = structlog.get_logger(__name__)

app = Server("ai-triage-agent")


# ── Tool manifest ──────────────────────────────────────────────────────────────

@app.list_tools()
async def handle_list_tools(_: ListToolsRequest) -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(
                name="fetch_test_logs",
                description=(
                    "Fetch CI test logs for a given run ID. "
                    "Supports GitHub Actions, local file, and mock backends."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "run_id": {
                            "type": "string",
                            "description": "GitHub Actions run ID or fixture key (e.g. 'cuda_oom')",
                        },
                        "backend": {
                            "type": "string",
                            "enum": ["github_actions", "local_file", "mock"],
                            "default": "github_actions",
                            "description": "Log source backend",
                        },
                    },
                    "required": ["run_id"],
                },
            ),
            Tool(
                name="get_git_diff",
                description=(
                    "Retrieve the unified git diff for a commit SHA. "
                    "Supports GitHub API, local git, and mock backends."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "commit_sha": {
                            "type": "string",
                            "description": "Full or short commit SHA",
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Absolute path to local git repo (for 'local' backend)",
                        },
                        "backend": {
                            "type": "string",
                            "enum": ["github_api", "local", "mock"],
                            "default": "github_api",
                        },
                    },
                    "required": ["commit_sha"],
                },
            ),
        ]
    )


# ── Tool dispatch ──────────────────────────────────────────────────────────────

@app.call_tool()
async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
    name = request.params.name
    args = request.params.arguments or {}

    log.info("mcp.tool_call", tool=name, args=args)

    try:
        if name == "fetch_test_logs":
            result = fetch_test_logs(**args)
        elif name == "get_git_diff":
            result = get_git_diff(**args)
        else:
            raise ValueError(f"Unknown tool: {name}")

        return CallToolResult(content=[TextContent(type="text", text=result)])

    except Exception as exc:
        log.exception("mcp.tool_error", tool=name)
        return CallToolResult(
            content=[TextContent(type="text", text=f"ERROR: {exc}")],
            isError=True,
        )


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("mcp_server.start", host=os.getenv("MCP_HOST", "127.0.0.1"))
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
