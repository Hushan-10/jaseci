"""Agent-level telemetry callback implementations."""

impl register_agent_callback(callback: Callable) -> None {
    if callback not in _agent_callbacks {
        _agent_callbacks.append(callback);
    }
}

impl _emit_agent_telemetry(record: dict) -> None {
    for callback in _agent_callbacks {
        try {
            callback(record);
        } except Exception as e {
            logger.debug(f"Agent telemetry callback error: {e}");
        }
    }
}
