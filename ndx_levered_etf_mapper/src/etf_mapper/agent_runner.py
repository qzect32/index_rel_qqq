from __future__ import annotations

from dotenv import load_dotenv

# This file is intentionally defensive: if you don't have the agents SDK installed,
# you can still run `python -m etf_mapper.cli refresh`.

def main() -> int:
    load_dotenv()

    try:
        from openai_agents import Agent, tool  # type: ignore
        from openai import OpenAI  # type: ignore
    except Exception:
        print("Agents SDK not installed. Install with: pip install -e '.[dev]'")
        return 2

    from .build import refresh_universe

    @tool
    def refresh(out_dir: str = "data") -> str:
        """Refresh the local ETF universe dataset and return output file paths."""
        outs = refresh_universe(out_dir)
        return "\n".join([f"{k}: {v}" for k, v in outs.items()])

    agent = Agent(
        name="UniverseRefresher",
        instructions=(
            "You help refresh a local dataset mapping Nasdaq-100 constituents to "
            "leveraged/inverse ETFs (including single-stock). "
            "Call the `refresh` tool once."
        ),
        tools=[refresh],
    )

    client = OpenAI()
    result = agent.run(client, "Update the universe and tell me where the files were written.")
    print(result.output_text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
