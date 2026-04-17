"""Reusable parallel tool-calling executor for byllm.

Provides a default scheduler that checks Tool.parallel_safe to decide
whether a batch of tool calls can run concurrently.  When all tools in
a batch have parallel_safe=True and the batch contains more than one
tool, uses ThreadPoolExecutor.  Otherwise falls back to sequential
execution that is byte-for-byte identical to BaseLLM._dispatch_tool_batch.

This module is framework-agnostic: it has NO dependency on jac-coder's
event system, thread-local state, or UI callbacks.

Usage (any byllm application):
    import from byllm.parallel { default_parallel_scheduler }
    llm.tool_scheduler = default_parallel_scheduler;
"""

import time;
import logging;

import from concurrent.futures { ThreadPoolExecutor, as_completed }
import from byllm.types { ToolCallResultMsg }

glob _MAX_WORKERS: int = 4,
     _plog = logging.getLogger("byllm.parallel");


"""Mark one or more tool functions as parallel-safe.

Sets the _parallel_safe attribute on each function so that Tool.postinit
picks it up automatically when wrapping the function.  Call this before
the first by-llm invocation that uses the tool.

Usage (Jac):
    with entry {
        mark_parallel_safe(get_temperature);
        mark_parallel_safe(get_humidity);
    }
"""
def mark_parallel_safe(func: Any) -> Any {
    setattr(func, '_parallel_safe', True);
    return func;
}


"""Sequential fallback — identical to BaseLLM._dispatch_tool_batch default.

Executes tool calls one-by-one in input order.  Used when the batch is
ineligible for parallel execution (single-tool, mixed parallel_safe,
or any dispatch-level error).

Returns list of (ToolCall, ToolCallResultMsg, duration_ms) in input order.
"""
def sequential_dispatch(tool_calls: list) -> list {
    results: list = [];
    for tc in tool_calls {
        t0 = time.time();
        msg = tc();
        ms = int((time.time() - t0) * 1000);
        results.append((tc, msg, ms));
    }
    return results;
}


"""Execute a single tool call in a worker thread.

On exception, wraps the error in a ToolCallResultMsg so the caller
always receives a valid result tuple (never raises).
"""
def _run_worker(tc: Any) -> tuple {
    t0 = time.time();
    try {
        msg = tc();
    } except Exception as e {
        msg = ToolCallResultMsg(
            content=f"Error (parallel worker): {str(e)}",
            tool_call_id=tc.call_id,
            name=tc.tool.get_name()
        );
    }
    ms = int((time.time() - t0) * 1000);
    return (tc, msg, ms);
}


"""Return a flat list of tool-name strings from a batch — used only for logging."""
def _tool_names(tool_calls: list) -> list {
    names: list = [];
    for tc in tool_calls {
        try {
            names.append(tc.tool.get_name());
        } except Exception {
            names.append("?");
        }
    }
    return names;
}


"""Execute tool calls in parallel via ThreadPoolExecutor.

Returns list of (ToolCall, ToolCallResultMsg, duration_ms) in INPUT ORDER
regardless of which worker finishes first.

On catastrophic pool failure, falls back to sequential_dispatch.
"""
def parallel_dispatch(tool_calls: list, max_workers: int = 4) -> list {
    n = len(tool_calls);
    workers = min(n, max_workers);

    try {
        futures: dict = {};
        raw: dict = {};
        with ThreadPoolExecutor(max_workers=workers) as pool {
            for (i, tc) in enumerate(tool_calls) {
                f = pool.submit(_run_worker, tc);
                futures[f] = i;
            }
            for f in as_completed(futures) {
                idx = futures[f];
                try {
                    raw[idx] = f.result();
                } except Exception as e {
                    tc_err = tool_calls[idx];
                    err_msg = ToolCallResultMsg(
                        content=f"Parallel dispatch error: {str(e)}",
                        tool_call_id=tc_err.call_id,
                        name=tc_err.tool.get_name()
                    );
                    raw[idx] = (tc_err, err_msg, 0);
                }
            }
        }
    } except Exception {
        _plog.warning("batch fallback=sequential reason=pool_error size=%d", n);
        return sequential_dispatch(tool_calls);
    }

    # Collect in input order.
    results: list = [];
    for i in range(n) {
        results.append(raw[i]);
    }
    return results;
}


"""Check whether every tool in the batch has parallel_safe=True."""
def _all_parallel_safe(tool_calls: list) -> bool {
    for tc in tool_calls {
        if not tc.tool.parallel_safe {
            return False;
        }
    }
    return True;
}


"""Default parallel scheduler for byllm applications.

Install on any BaseLLM instance to enable automatic parallel dispatch
for tool batches where every tool is marked parallel_safe=True.

Decision tree (first matching rule wins):
  1. Batch size <= 1          → sequential  (no parallelism benefit)
  2. Any tool has parallel_safe=False → sequential (safety)
  3. All tools parallel_safe=True     → parallel via ThreadPoolExecutor

Returns list of (ToolCall, ToolCallResultMsg, duration_ms) in INPUT ORDER.
"""
def default_parallel_scheduler(tool_calls: list) -> list {
    n = len(tool_calls);
    names = _tool_names(tool_calls);

    # Guard: no benefit from a pool with one or zero workers.
    if n <= 1 {
        _plog.debug("batch dispatch=sequential reason=single_or_empty size=%d", n);
        return sequential_dispatch(tool_calls);
    }

    # Guard: any non-parallel-safe tool forces the whole batch sequential.
    if not _all_parallel_safe(tool_calls) {
        _plog.info(
            "batch dispatch=sequential reason=mixed_safety size=%d tools=%s", n, names
        );
        return sequential_dispatch(tool_calls);
    }

    # --- Parallel path ---
    workers = min(n, _MAX_WORKERS);
    _plog.info(
        "batch dispatch=parallel size=%d workers=%d tools=%s", n, workers, names
    );
    t_batch = time.time();

    results = parallel_dispatch(tool_calls, max_workers=workers);

    wall_ms = int((time.time() - t_batch) * 1000);
    sum_ms = 0;
    for (_, _, ms) in results {
        sum_ms += ms;
    }
    speedup = round(sum_ms / wall_ms, 2) if wall_ms > 0 else 0.0;
    _plog.info(
        "batch done=parallel size=%d workers=%d wall_ms=%d sum_tool_ms=%d speedup=%sx",
        n,
        workers,
        wall_ms,
        sum_ms,
        speedup
    );

    return results;
}
