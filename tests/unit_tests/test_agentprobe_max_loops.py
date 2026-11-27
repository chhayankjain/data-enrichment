"""Regression test: max_loops bounds research, whatever the model does.

Uses AgentProbe's scripted model (test-only dependency; no model API calls or
credentials): pip install "agentprobe-toolkit[langchain]>=1.6"
The page to scrape is served by a local HTTP server on 127.0.0.1.

On main both cases fail: the model keeps being called until the 40-call budget runs
out. With the fix in fix/robust-research-loop both pass.
"""

import asyncio
import http.server
import importlib
import threading

import pytest
from agentprobe.testing import probe, script

# The package re-exports the compiled `graph`, which shadows the submodule name.
graph_mod = importlib.import_module("enrichment_agent.graph")
tools_mod = importlib.import_module("enrichment_agent.tools")

INPUT = {
    "topic": "Example Corp",
    "extraction_schema": {"type": "object", "properties": {"founded": {"type": "string"}}},
}


class _Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        data = b"<p>Founded 1999.</p>"
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def page_url():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/about"
    server.shutdown()


@pytest.mark.parametrize("behaviour", ["keeps-researching", "never-calls-a-tool"])
def test_max_loops_bounds_the_run(monkeypatch, page_url, behaviour):
    step = (
        script.tool_call("scrape_website", {"url": page_url})
        if behaviour == "keeps-researching"
        else script.reply("I believe it was 1999.")
    )
    with probe() as p:
        model = p.model(script.repeat(step), max_calls=40).langchain()
        monkeypatch.setattr(graph_mod, "init_model", lambda config=None: model)
        monkeypatch.setattr(tools_mod, "init_model", lambda config=None: model)
        r = p.run(
            lambda _: asyncio.run(
                graph_mod.graph.ainvoke(INPUT, config={"configurable": {"max_loops": 2}})
            ),
            "topic",
            timeout=30,
        )
        r.assert_no_crash()
        # max_loops=2 allows 2 agent calls (the requests that offer tools). The only
        # other model calls are scrape_website's page summaries, one per page scraped.
        agent_calls = [c for c in r.model_calls if c.tools_offered]
        assert len(agent_calls) <= 2, f"{len(agent_calls)} agent calls with max_loops=2"
        r.assert_model_calls_at_most(3)
