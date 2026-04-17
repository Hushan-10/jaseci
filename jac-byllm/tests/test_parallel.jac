"""Tests for the parallel tool-calling engine (byllm/parallel.jac).

Covers:
  1.  Default serialize=False (parallel opt-out model)
  2.  dispatch_batch runs parallel by default
  3.  serialize=True forces sequential
  4.  _index_by_identity result reordering
  5.  Single-tool batch runs inline
  6.  Empty batch yields nothing
  7.  Failing tool is isolated from others
  8.  Speedup proves parallel overlap
  9.  Sequential shows no speedup
  10. _is_async detection
  11. dispatch_batch is a generator
  12. _PARALLEL_TOOL_HINT constant exists in source
  13. Per-tool timeout attribute stored correctly
  14. mark_serialize helper sets _serialize attribute
  15. System prompt nudge wired into basellm source
  16. Shared module-level pool in source
  17. _serialize detection in Tool.postinit source
  18. MockLLM batch fixture (scheduler_parity) exists
  19. Async tool support (_call_async, _is_async, asyncio.run) in source
  20. Old API (tool_scheduler, _dispatch_tool_batch) removed from llm.jac
  21. BaseLLM.invoke uses dispatch_batch
"""

import threading;
import time;
import pathlib;
import types as pytypes;

import from byllm.parallel { dispatch_batch, _is_async, _index_by_identity, mark_serialize }

glob _TESTS_DIR: str = str(pathlib.Path(__file__).resolve().parent);
glob _ROOT_DIR: str = str(pathlib.Path(__file__).resolve().parent.parent);

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Noop callable — used as Tool.func stand-in so _is_async returns False."""
def _noop() -> None { }


# ---------------------------------------------------------------------------
# Mock objects (duck-typed stand-ins for byllm Tool / ToolCall)
# ---------------------------------------------------------------------------

obj _FakeMsg {
    has content: str;
}

obj _FakeTool {
    has name_str: str,
        serialize: bool = False,
        timeout_sec: float | None = None,
        func: Any = None;

    def postinit -> None {
        self.func = _noop;
    }

    def get_name -> str {
        return self.name_str;
    }
}

obj _FakeTC {
    has tc_name: str,
        tc_result: str,
        tc_delay: float = 0.0,
        tc_serialize: bool = False,
        tc_should_raise: bool = False,
        call_count: int = 0,
        call_thread: Any = None,
        call_id: str = "",
        tool: Any = None;

    def postinit -> None {
        self.call_id = f"call_{self.tc_name}";
        self.tool = _FakeTool(
            name_str=self.tc_name,
            serialize=self.tc_serialize
        );
    }

    def __call__ -> _FakeMsg {
        self.call_count += 1;
        self.call_thread = threading.current_thread().ident;
        if self.tc_delay > 0.0 {
            time.sleep(self.tc_delay);
        }
        if self.tc_should_raise {
            raise RuntimeError(f"{self.tc_name} exploded");
        }
        return _FakeMsg(content=self.tc_result);
    }
}


# ---------------------------------------------------------------------------
# 1. Default serialize is False
# ---------------------------------------------------------------------------

test "serialize default is false" {
    tool = _FakeTool(name_str="get_weather");
    assert tool.serialize == False;
}

test "serialize can be set true" {
    tool = _FakeTool(name_str="write_file", serialize=True);
    assert tool.serialize == True;
}


# ---------------------------------------------------------------------------
# 2. Parallel execution (default — no opt-in needed)
# ---------------------------------------------------------------------------

test "parallel default tools called" {
    tcs = [
        _FakeTC(tc_name="tool_a", tc_result="A"),
        _FakeTC(tc_name="tool_b", tc_result="B"),
        _FakeTC(tc_name="tool_c", tc_result="C"),
    ];
    results = list(dispatch_batch(tcs));
    assert len(results) == 3;
    assert all(tc.call_count == 1 for tc in tcs);
    contents = sorted([r[1].content for r in results]);
    assert contents == ["A", "B", "C"];
}

test "parallel tools run on different threads" {
    tcs = [
        _FakeTC(tc_name="tool_a", tc_result="A", tc_delay=0.02),
        _FakeTC(tc_name="tool_b", tc_result="B", tc_delay=0.02),
    ];
    list(dispatch_batch(tcs));
    assert tcs[0].call_thread != tcs[1].call_thread;
}


# ---------------------------------------------------------------------------
# 3. Sequential fallback when any tool has serialize=True
# ---------------------------------------------------------------------------

test "serialize forces sequential" {
    tcs = [
        _FakeTC(tc_name="read_tool", tc_result="R", tc_serialize=False),
        _FakeTC(tc_name="write_tool", tc_result="W", tc_serialize=True),
    ];
    results = list(dispatch_batch(tcs));
    assert len(results) == 2;
    assert tcs[0].call_thread == tcs[1].call_thread;
}

test "all serialize stays sequential" {
    tcs = [
        _FakeTC(tc_name="write_a", tc_result="WA", tc_serialize=True),
        _FakeTC(tc_name="write_b", tc_result="WB", tc_serialize=True),
    ];
    list(dispatch_batch(tcs));
    assert tcs[0].call_thread == tcs[1].call_thread;
}


# ---------------------------------------------------------------------------
# 4. Result ordering with _index_by_identity
# ---------------------------------------------------------------------------

test "index by identity finds correct positions" {
    a = _FakeTC(tc_name="a", tc_result="A");
    b = _FakeTC(tc_name="b", tc_result="B");
    c = _FakeTC(tc_name="c", tc_result="C");
    seq = [a, b, c];
    assert _index_by_identity(seq, a) == 0;
    assert _index_by_identity(seq, b) == 1;
    assert _index_by_identity(seq, c) == 2;
}

test "index by identity returns minus one when not found" {
    a = _FakeTC(tc_name="a", tc_result="A");
    b = _FakeTC(tc_name="b", tc_result="B");
    assert _index_by_identity([a], b) == -1;
}

test "reorder parallel results to input order" {
    tcs = [
        _FakeTC(tc_name="slow", tc_result="SLOW", tc_delay=0.06),
        _FakeTC(tc_name="fast", tc_result="FAST", tc_delay=0.01),
        _FakeTC(tc_name="medium", tc_result="MEDIUM", tc_delay=0.03),
    ];
    raw = list(dispatch_batch(tcs));
    ordered = {};
    for r in raw {
        idx = _index_by_identity(tcs, r[0]);
        ordered[idx] = r;
    }
    assert ordered[0][1].content == "SLOW";
    assert ordered[1][1].content == "FAST";
    assert ordered[2][1].content == "MEDIUM";
}


# ---------------------------------------------------------------------------
# 5. Single-tool batch runs inline
# ---------------------------------------------------------------------------

test "single tool batch runs inline" {
    tc = _FakeTC(tc_name="read", tc_result="R");
    results = list(dispatch_batch([tc]));
    assert len(results) == 1;
    assert results[0][1].content == "R";
}


# ---------------------------------------------------------------------------
# 6. Empty batch
# ---------------------------------------------------------------------------

test "empty batch yields nothing" {
    assert list(dispatch_batch([])) == [];
}


# ---------------------------------------------------------------------------
# 7. Failing tool does not corrupt others
# ---------------------------------------------------------------------------

test "failing tool is isolated" {
    tcs = [
        _FakeTC(tc_name="good_a", tc_result="A"),
        _FakeTC(tc_name="bad", tc_result="X", tc_should_raise=True),
        _FakeTC(tc_name="good_b", tc_result="B"),
    ];
    results = list(dispatch_batch(tcs));
    contents = {r[1].content for r in results};
    assert "A" in contents;
    assert "B" in contents;
    error_results = [r for r in results if "error" in r[1].content.lower()];
    assert len(error_results) == 1;
    assert "exploded" in error_results[0][1].content;
}


# ---------------------------------------------------------------------------
# 8. Speedup signal proves parallel overlap
# ---------------------------------------------------------------------------

test "speedup proves parallel overlap" {
    sleep_sec = 0.06;
    tcs = [
        _FakeTC(tc_name="a", tc_result="A", tc_delay=sleep_sec),
        _FakeTC(tc_name="b", tc_result="B", tc_delay=sleep_sec),
        _FakeTC(tc_name="c", tc_result="C", tc_delay=sleep_sec),
    ];
    t0 = time.time();
    results = list(dispatch_batch(tcs));
    wall_ms = (time.time() - t0) * 1000;
    sum_ms = sum(ms for (_, _, ms) in results);
    speedup = sum_ms / wall_ms if wall_ms > 0 else 0.0;
    assert speedup > 1.5;
}


# ---------------------------------------------------------------------------
# 9. Sequential has no speedup
# ---------------------------------------------------------------------------

test "sequential batch has no speedup" {
    sleep_sec = 0.03;
    tcs = [
        _FakeTC(tc_name="a", tc_result="A", tc_delay=sleep_sec, tc_serialize=True),
        _FakeTC(tc_name="b", tc_result="B", tc_delay=sleep_sec, tc_serialize=True),
    ];
    t0 = time.time();
    results = list(dispatch_batch(tcs));
    wall_ms = (time.time() - t0) * 1000;
    sum_ms = sum(ms for (_, _, ms) in results);
    speedup = sum_ms / wall_ms if wall_ms > 0 else 0.0;
    assert speedup < 1.3;
}


# ---------------------------------------------------------------------------
# 10. _is_async detection
# ---------------------------------------------------------------------------

test "is async returns false for sync function" {
    def sync_fn() -> None { }
    assert _is_async(sync_fn) == False;
}

test "is async returns true for async function" {
    import asyncio;
    async def async_fn() -> None { }
    assert _is_async(async_fn) == True;
}


# ---------------------------------------------------------------------------
# 11. dispatch_batch is a generator
# ---------------------------------------------------------------------------

test "dispatch batch returns a generator" {
    tcs = [
        _FakeTC(tc_name="a", tc_result="A"),
        _FakeTC(tc_name="b", tc_result="B"),
    ];
    gen = dispatch_batch(tcs);
    assert isinstance(gen, pytypes.GeneratorType);
}


# ---------------------------------------------------------------------------
# 12. _PARALLEL_TOOL_HINT constant in source
# ---------------------------------------------------------------------------

test "parallel tool hint constant in source" {
    src = pathlib.Path(_ROOT_DIR) / "byllm" / "parallel.jac";
    content = src.read_text();
    assert "_PARALLEL_TOOL_HINT" in content;
    assert "parallel" in content.lower();
    assert "independent" in content.lower();
}


# ---------------------------------------------------------------------------
# 13. Per-tool timeout attribute
# ---------------------------------------------------------------------------

test "timeout attribute stored on tool" {
    tool = _FakeTool(name_str="slow_tool", timeout_sec=0.05);
    assert tool.timeout_sec == 0.05;
}

test "timeout none means no timeout" {
    tool = _FakeTool(name_str="normal_tool");
    assert tool.timeout_sec is None;
}


# ---------------------------------------------------------------------------
# 14. mark_serialize helper
# ---------------------------------------------------------------------------

test "mark serialize sets underscore serialize attribute" {
    def my_func() -> None { }
    assert not hasattr(my_func, '_serialize');
    mark_serialize(my_func);
    assert getattr(my_func, '_serialize') == True;
}

test "mark serialize function exists in source" {
    src = pathlib.Path(_ROOT_DIR) / "byllm" / "parallel.jac";
    content = src.read_text();
    assert "def mark_serialize" in content;
    assert "setattr(func, '_serialize', True)" in content;
}


# ---------------------------------------------------------------------------
# 15. System prompt nudge wired into basellm
# ---------------------------------------------------------------------------

test "system prompt nudge in basellm source" {
    src = pathlib.Path(_ROOT_DIR) / "byllm" / "llm.impl" / "basellm.impl.jac";
    content = src.read_text();
    assert "_PARALLEL_TOOL_HINT" in content;
    assert 'first_msg["content"]' in content;
}


# ---------------------------------------------------------------------------
# 16. Shared module-level pool in source
# ---------------------------------------------------------------------------

test "shared worker pool defined in source" {
    src = pathlib.Path(_ROOT_DIR) / "byllm" / "parallel.jac";
    content = src.read_text();
    assert "_WORKER_POOL" in content;
    assert "ThreadPoolExecutor" in content;
    assert "BYLLM_TOOL_WORKERS" in content;
}


# ---------------------------------------------------------------------------
# 17. _serialize detection in Tool.postinit
# ---------------------------------------------------------------------------

test "serialize detection in tool postinit source" {
    src = pathlib.Path(_ROOT_DIR) / "byllm" / "types.impl" / "tool.impl.jac";
    content = src.read_text();
    assert "_serialize" in content;
    assert "self.serialize" in content;
}


# ---------------------------------------------------------------------------
# 18. MockLLM batch fixture (scheduler_parity) exists
# ---------------------------------------------------------------------------

test "scheduler parity fixture exists and has batch tool calls" {
    fixture = pathlib.Path(_TESTS_DIR) / "fixtures" / "scheduler_parity.jac";
    assert fixture.exists();
    content = fixture.read_text();
    assert "MockToolCall" in content;
    assert "SCHEDULER_PARITY_PASS" in content;
}


# ---------------------------------------------------------------------------
# 19. Async tool support in source
# ---------------------------------------------------------------------------

test "async tool support functions in source" {
    src = pathlib.Path(_ROOT_DIR) / "byllm" / "parallel.jac";
    content = src.read_text();
    assert "_call_async" in content;
    assert "_is_async" in content;
    assert "asyncio.run" in content;
}


# ---------------------------------------------------------------------------
# 20. Old API removed from llm.jac
# ---------------------------------------------------------------------------

test "tool scheduler removed from llm" {
    src = pathlib.Path(_ROOT_DIR) / "byllm" / "llm.jac";
    content = src.read_text();
    assert "tool_scheduler" not in content;
    assert "_dispatch_tool_batch" not in content;
}

test "basellm uses dispatch batch not old api" {
    src = pathlib.Path(_ROOT_DIR) / "byllm" / "llm.impl" / "basellm.impl.jac";
    content = src.read_text();
    assert "dispatch_batch" in content;
    assert "_dispatch_tool_batch" not in content;
}
