"""In-memory MCP smoke test — no frontend or subprocess needed.

Usage (from repo root, with venv active):
    python mcp_server/test_mcp.py
    python mcp_server/test_mcp.py --send   # actually call Gmail (needs credentials.json)
"""

import argparse
import asyncio
import os
import sys

# Ensure repo root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import Client
from mcp_server.server import mcp


async def list_tools(client: Client) -> None:
    await client.ping()
    tools = await client.list_tools()
    print("=== MCP tools ===")
    for tool in tools:
        print(f"  {tool.name}")
        print(f"    {tool.description or '(no description)'}")


async def call_send_email(client: Client) -> None:
    result = await client.call_tool(
        "send_confirmation_email",
        {
            "patient_email": "you@example.com",
            "patient_name": "Test Patient",
            "doctor_name": "Dr. Smith",
            "appointment_date": "2026-08-15",
            "appointment_time": "10:00",
        },
    )
    print("=== send_confirmation_email result ===")
    print(result.data)


async def main(send: bool) -> None:
    async with Client(mcp) as client:
        await list_tools(client)
        if send:
            creds_path = os.path.join(os.getcwd(), "credentials.json")
            if not os.path.exists(creds_path):
                print("\nSkipping send: credentials.json not found in repo root.")
                print("Add Gmail OAuth credentials there, then re-run with --send.")
                return
            print("\nCalling send_confirmation_email (opens browser on first auth)...")
            await call_send_email(client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Hospital MCP server in-memory")
    parser.add_argument(
        "--send",
        action="store_true",
        help="Call send_confirmation_email (requires credentials.json + token.json)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.send))
