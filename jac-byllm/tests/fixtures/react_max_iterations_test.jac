import from byllm.lib { MockLLM, MockToolCall }

glob tool_calls = {"wind": 0, "unit": 0};

glob llm: MockLLM = MockLLM(
        model_name="mockllm",
        config={
            "outputs": [
                 # Iteration 1: model asks to call tool
                MockToolCall(tool=get_live_wind_speed, args={"city": "Puttalam"}),

                # Forced final-answer call (tools disabled) -> returns plain text
                "FINAL_REPORT",

                # Iteration 2: model WOULD ask to call tool again (but max_react_iterations=1 should stop before this runs)
                MockToolCall(tool=get_speed_unit, args={})
        
        ],
        "show_params": True,
        "verbose": True
    })

def get_live_wind_speed(city: str) -> str {
    tool_calls["wind"] += 1;
    print(f"get_live_wind_speed called for {city}");
    return f"The current wind speed in {city} is about 18-22 km/h.";
}

sem get_live_wind_speed = """Get the current wind speed of a given city.""";

def get_speed_unit() -> str {
    tool_calls["unit"] += 1;
    print("get_speed_unit called");
    return "km/h";
}
sem get_speed_unit = "Return the unit used for wind speed reporting.";

def get_wind_speed_report(city: str) -> str by llm(
    method="ReAct",
    tools=[get_live_wind_speed, get_speed_unit],
    max_react_iterations=1
);

glob result = get_wind_speed_report("Puttalam");

with entry {
    print("RESULT: " + result);
    print("WIND_TOOL_CALLS: " + str(tool_calls["wind"]));
    print("UNIT_TOOL_CALLS: " + str(tool_calls["unit"]));
}