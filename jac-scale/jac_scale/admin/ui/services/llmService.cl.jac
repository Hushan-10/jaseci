"""LLM Telemetry service functions for admin dashboard."""

cl {
    import from .api { get }

    async def:pub getLLMSummary(token: str) -> dict {
        return await get("/admin/llm/telemetry/summary", token);
    }

    async def:pub getLLMTraces(token: str) -> dict {
        return await get("/admin/llm/telemetry/traces?limit=50&offset=0", token);
    }

    async def:pub getLLMTraceDetail(token: str, invocationId: str) -> dict {
        return await get(
            "/admin/llm/telemetry/traces/" + encodeURIComponent(invocationId), token
        );
    }

    async def:pub getLLMFilters(token: str) -> dict {
        return await get("/admin/llm/telemetry/filters", token);
    }
}
