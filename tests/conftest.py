import asyncio
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "synthetic_ulsd.csv"


@pytest.fixture(scope="session")
def sample_csv():
    if not SAMPLE.exists():
        from scripts.generate_sample_data import generate  # pragma: no cover

        generate().to_csv(SAMPLE, index=False)
    return str(SAMPLE)


class Caller:
    """Synchronous helper around the FastMCP in-memory client."""

    def __init__(self):
        from fastmcp import Client

        from hydromcp.server import mcp

        self.client = Client(mcp)
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.client.__aenter__())

    def __call__(self, name, **args):
        res = self.loop.run_until_complete(self.client.call_tool(name, args))
        if res.structured_content is not None:
            sc = res.structured_content
            return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
        return json.loads(res.content[0].text)

    def close(self):
        self.loop.run_until_complete(self.client.__aexit__(None, None, None))
        self.loop.close()


@pytest.fixture(scope="session")
def call():
    c = Caller()
    yield c
    c.close()
