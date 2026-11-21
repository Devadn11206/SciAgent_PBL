"""Standalone Stdio/CLI Runner for SciAgent FastMCP Server.

Can be added to Claude Desktop / Cursor / VS Code `mcpServers`:
{
  "mcpServers": {
    "sciagent": {
      "command": "python",
      "args": ["-m", "backend.app.mcp.cli"]
    }
  }
}
"""

from __future__ import annotations

import asyncio
from backend.app.mcp.sciagent_mcp import mcp_server


def main():
    mcp_server.run()


if __name__ == "__main__":
    main()
