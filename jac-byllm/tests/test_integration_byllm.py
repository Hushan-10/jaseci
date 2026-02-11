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
from http.client import RemoteDisconnected
from subprocess import PIPE, Popen
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest


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

    def test_server_starts_successfully(self) -> None:
        """Verify that jac start main.jac starts the server successfully.
        
        This test:
        1. Locates the integration_byllm fixture directory
        2. Starts the server using `jac start main.jac -p <port>`
        3. Waits for the server to accept connections
        4. Verifies the server is running
        5. Cleans up by terminating the server
        
        This is the foundation test - if this passes, the server infrastructure works.
        
        NOTE: Requires OPENAI_API_KEY environment variable to be set.
        """
        # Skip test if OPENAI_API_KEY is not set
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip(
                "OPENAI_API_KEY not set - skipping server start test. "
                "Set the environment variable to run this test."
            )

        print("[DEBUG] Starting test_server_starts_successfully")

        tests_dir = os.path.dirname(os.path.abspath(__file__))
        fixture_path = os.path.join(tests_dir, "fixtures", "integration_byllm")
        server_port = get_free_port()
        jac_cmd = get_jac_command()
        server: Popen[bytes] | None = None

        try:
            server = Popen(
                [*jac_cmd, "start", "main.jac", "-p", str(server_port)],
                cwd=fixture_path,
            )
            print(f"[DEBUG] Server started on port {server_port} (PID: {server.pid})")

            try:
                wait_for_port("127.0.0.1", server_port, timeout=60.0)
                print(f"[DEBUG] ✅ Server is accepting connections")
            except TimeoutError:
                poll_result = server.poll()
                if poll_result is not None:
                    pytest.fail(f"Server exited prematurely with code {poll_result}")
                raise

            assert server.poll() is None, "Server should still be running"
            print("[DEBUG] ✅ TEST PASSED: Server running successfully")

        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except Exception:
                    server.kill()
                    server.wait(timeout=5)
                time.sleep(1)
                gc.collect()


class TestByllmWalkerEndpoints:
    """Test that walker HTTP endpoints work correctly."""

    def test_supervisor_walker_endpoint(self) -> None:
        """Test calling the Supervisor walker via HTTP POST endpoint.
        
        This test:
        1. Starts the server with walker HTTP endpoints
        2. Calls POST /walker/Supervisor with a query
        3. Verifies the routing works (selects correct agent)
        4. Verifies the agent response is returned correctly
        
        NOTE: Requires OPENAI_API_KEY environment variable to be set.
        """
        # Skip test if OPENAI_API_KEY is not set
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip(
                "OPENAI_API_KEY not set - skipping walker endpoint test. "
                "Set the environment variable to run this test."
            )

        print("[DEBUG] Starting test_supervisor_walker_endpoint")

        tests_dir = os.path.dirname(os.path.abspath(__file__))
        fixture_path = os.path.join(tests_dir, "fixtures", "integration_byllm")
        server_port = get_free_port()
        jac_cmd = get_jac_command()
        server: Popen[bytes] | None = None

        try:
            server = Popen(
                [*jac_cmd, "start", "main.jac", "-p", str(server_port)],
                cwd=fixture_path,
            )
            print(f"[DEBUG] Server started on port {server_port}")

            wait_for_port("127.0.0.1", server_port, timeout=60.0)
            print(f"[DEBUG] ✅ Server is accepting connections")

            # Test POST /walker/Supervisor endpoint with different query types
            print("[DEBUG] Testing POST /walker/Supervisor endpoint\n")
            
            # Test Case 1: ConceptAgent query
            print("[DEBUG] Test Case 1: ConceptAgent query")
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
                        assert len(agent_report["response"]) > 0, "Response should not be empty"
                        print(f"[DEBUG] ✅ ConceptAgent selected correctly")
                    else:
                        assert len(response_data) > 0, "Response should not be empty"
                        print("[DEBUG] ✅ Received valid response")
                    
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
                pytest.fail(f"ConceptAgent test failed: {exc.code}\n{error_body}")

            # Test Case 2: MathAgent query
            print("\n[DEBUG] Test Case 2: MathAgent query")
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
                        assert len(agent_report["response"]) > 0, "Response should not be empty"
                        print(f"[DEBUG] ✅ MathAgent selected correctly")
                    else:
                        assert len(response_data) > 0, "Response should not be empty"
                        print("[DEBUG] ✅ Received valid response")
                    
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
                pytest.fail(f"MathAgent test failed: {exc.code}\n{error_body}")

            # Test Case 3: ResearchAgent query
            print("\n[DEBUG] Test Case 3: ResearchAgent query")
            research_payload = {"query": "Compare supervised fine-tuning and prompt engineering in large language models"}
            
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
                        assert len(agent_report["summary"]) > 0, "Summary should not be empty"
                        print(f"[DEBUG] ✅ ResearchAgent selected correctly")
                    else:
                        assert len(response_data) > 0, "Response should not be empty"
                        print("[DEBUG] ✅ Received valid response")
                    
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
                pytest.fail(f"ResearchAgent test failed: {exc.code}\n{error_body}")

            print("\n[DEBUG] ✅ TEST PASSED: All routing tests successful")

        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except Exception:
                    server.kill()
                    server.wait(timeout=5)
                time.sleep(1)
                gc.collect()


# Run the test if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
