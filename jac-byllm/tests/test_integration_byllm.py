"""Integration tests for byllm multi-agent application.

This test suite validates:
- jac start command works properly
- Server starts and accepts connections
- Walker HTTP endpoints are accessible
- Supervisor routing works correctly
- Agent responses are correct
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import socket
import sys
import time
from subprocess import Popen
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from typing import Generator


@pytest.fixture(scope="module")
def byllm_server() -> Generator[tuple[Popen[bytes], int], None, None]:
    """Start the byllm server once for all tests in this module.
    
    Yields:
        Tuple of (server_process, port_number)
    """
    # Skip all tests if OPENAI_API_KEY is not set
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip(
            "OPENAI_API_KEY not set - skipping all integration tests. "
            "Set the environment variable to run these tests."
        )
    
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_path = os.path.join(tests_dir, "fixtures", "integration_byllm")
    server_port = get_free_port()
    jac_cmd = get_jac_command()
    
    server = Popen(
        [*jac_cmd, "start", "main.jac", "-p", str(server_port)],
        cwd=fixture_path,
    )
    
    try:
        wait_for_port("127.0.0.1", server_port, timeout=60.0)
        
        # Yield server and port to tests
        yield server, server_port
        
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:
            server.kill()
            server.wait(timeout=5)
        time.sleep(1)
        gc.collect()


def get_free_port() -> int:
    """Get a free port by binding to port 0 and releasing it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def get_jac_command() -> list[str]:
    """Get the jac command with proper path handling."""
    jac_path = shutil.which("jac")
    if jac_path:
        return [jac_path]
    return [sys.executable, "-m", "jaclang"]


def wait_for_port(host: str, port: int, timeout: float = 60.0) -> None:
    """Block until a TCP port is accepting connections or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect((host, port))
                return
            except OSError:
                time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


class TestByllmServerStart:
    """Test that jac start command works for the byllm integration app."""

    def test_server_starts_successfully(
        self, byllm_server: tuple[Popen[bytes], int]
    ) -> None:
        """Verify that jac start main.jac starts the server successfully.

        This test verifies:
        1. Server process is running
        2. Server port is accepting connections
        3. Server is stable (hasn't crashed)

        NOTE: Server is started once by the byllm_server fixture for all tests.
        """
        server, server_port = byllm_server

        # Verify server process is still alive
        assert server.poll() is None, "Server should be running"

        # Verify port is still accepting connections
        try:
            wait_for_port("127.0.0.1", server_port, timeout=5.0)
        except TimeoutError:
            pytest.fail("Server port is not accepting connections")


class TestByllmWalkerEndpoints:
    """Test that walker HTTP endpoints work correctly."""

    def test_supervisor_walker_endpoint(
        self, byllm_server: tuple[Popen[bytes], int]
    ) -> None:
        """Test calling the Supervisor walker via HTTP POST endpoint.

        This test:
        1. Calls POST /walker/Supervisor with different queries
        2. Verifies the routing works (selects correct agent)
        3. Verifies the agent response is returned correctly

        NOTE: Uses the shared byllm_server fixture that was already started.
        """
        server, server_port = byllm_server

        # Test Case 1: ConceptAgent query
        concept_payload = {"query": "Explain machine learning in simple terms"}

        req = Request(
            f"http://127.0.0.1:{server_port}/walker/Supervisor",
            data=json.dumps(concept_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=90) as resp:
                response_body = resp.read().decode("utf-8", errors="ignore")
                assert resp.status == 200, f"Expected 200, got {resp.status}"
                response_data = json.loads(response_body)

                if "reports" in response_data:
                    reports = response_data["reports"]
                    assert len(reports) > 0, "Should have at least one report"
                    agent_report = reports[0]
                    assert agent_report["agent"] == "ConceptAgent", (
                        f"Expected ConceptAgent, got {agent_report.get('agent')}"
                    )
                    assert len(agent_report["response"]) > 0, (
                        "Response should not be empty"
                    )
                else:
                    assert len(response_data) > 0, "Response should not be empty"

        except HTTPError as exc:
            error_body = (
                exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            )
            pytest.fail(f"ConceptAgent test failed: {exc.code}\n{error_body}")

        # Test Case 2: MathAgent query
        math_payload = {"query": "What is (15 + 5) * 2 - (10 / 2)?"}

        req = Request(
            f"http://127.0.0.1:{server_port}/walker/Supervisor",
            data=json.dumps(math_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=90) as resp:
                response_body = resp.read().decode("utf-8", errors="ignore")
                assert resp.status == 200, f"Expected 200, got {resp.status}"
                response_data = json.loads(response_body)

                if "reports" in response_data:
                    reports = response_data["reports"]
                    assert len(reports) > 0, "Should have at least one report"
                    agent_report = reports[0]
                    assert agent_report["agent"] == "MathAgent", (
                        f"Expected MathAgent, got {agent_report.get('agent')}"
                    )
                    assert len(agent_report["response"]) > 0, (
                        "Response should not be empty"
                    )
                else:
                    assert len(response_data) > 0, "Response should not be empty"

        except HTTPError as exc:
            error_body = (
                exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            )
            pytest.fail(f"MathAgent test failed: {exc.code}\n{error_body}")

        # Test Case 3: ResearchAgent query
        research_payload = {
            "query": "Compare supervised fine-tuning and prompt engineering in large language models"
        }

        req = Request(
            f"http://127.0.0.1:{server_port}/walker/Supervisor",
            data=json.dumps(research_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=90) as resp:
                response_body = resp.read().decode("utf-8", errors="ignore")
                assert resp.status == 200, f"Expected 200, got {resp.status}"
                response_data = json.loads(response_body)

                if "reports" in response_data:
                    reports = response_data["reports"]
                    assert len(reports) > 0, "Should have at least one report"
                    agent_report = reports[0]
                    assert agent_report["agent"] == "ResearchAgent", (
                        f"Expected ResearchAgent, got {agent_report.get('agent')}"
                    )
                    # ResearchAgent returns different fields
                    assert "summary" in agent_report, "Should have summary field"
                    assert len(agent_report["summary"]) > 0, (
                        "Summary should not be empty"
                    )
                else:
                    assert len(response_data) > 0, "Response should not be empty"

        except HTTPError as exc:
            error_body = (
                exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
            )
            pytest.fail(f"ResearchAgent test failed: {exc.code}\n{error_body}")


# Run the test if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
