"""Agent-level telemetry callback registry.

byllm emits events after each agent invocation completes.
Consumers (e.g., jac-scale) register callbacks to receive these events.
No data is stored in byllm — this is purely a publish mechanism.
"""
import from typing { Callable }
import logging;

glob _agent_callbacks: list = [],
     logger = logging.getLogger(__name__);

"""Register a callback to receive agent invocation completion events."""
def:pub register_agent_callback(callback: Callable) -> None;

"""Emit an agent-level telemetry record to all registered callbacks."""
def _emit_agent_telemetry(record: dict) -> None;
