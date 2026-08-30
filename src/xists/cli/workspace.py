"""Workspace initialization, MCP launcher, and version CLI handlers."""

from __future__ import annotations

import argparse
import sys

from xists import __version__
from xists.workspace import (
    initialize_workspace,
    populate_demo_workspace,
    workspace_root,
)


def version(args: argparse.Namespace) -> int:
    print(f"xists {__version__}")
    return 0


def mcp(args: argparse.Namespace) -> int:
    """Start the optional MCP server without writing protocol data to stdout."""
    from xists.mcp_server import MCPNotInstalledError, MCPStartupError, run_server

    try:
        run_server(args.index)
    except (MCPNotInstalledError, MCPStartupError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


def workspace_init(args: argparse.Namespace) -> int:
    root = workspace_root()
    demo_mode = getattr(args, "demo", False)
    created_root, created_env_file = initialize_workspace(root)
    demo_info = None
    if demo_mode:
        try:
            demo_info = populate_demo_workspace(root, force=getattr(args, "force", False))
        except FileExistsError as error:
            print(f"Note: {error}", file=sys.stderr)

    status = "initialized" if created_root or created_env_file or demo_info else "already exists"
    if demo_mode:
        status += " (demo mode)"
    print("Workspace")
    print(f"  Location  {root}")
    print(f"  Status    {status}")
    if created_env_file:
        print(f"  Config    {root / '.env'}")
    if demo_info:
        print(
            f"  Records   {demo_info.get('records_path')} ({demo_info.get('records_count', 200)} starter repos)"
        )
        print(f"  Index     {demo_info.get('index_path')} (v4 dual-file binary)")
    print("\nNext steps")
    if demo_mode:
        print('  1. Try search: xists search "fast web framework"')
        print('  2. Try offline demo: xists search --demo "vector database"')
        print(f"  3. Configure API keys in {root / '.env'} when ready to index your own repos")
    else:
        print(f"  1. Edit {root / '.env'}")
        print("  2. Run xists doctor")
    return 0


__all__ = [
    "mcp",
    "version",
    "workspace_init",
]
