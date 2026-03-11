"""Post-initialization hook for Model.

Sets up proxy mode and mock delegation based on config.
Configures logging when using real LLM backends.
"""
impl Model.postinit -> None {
    # Check instance config first, then fall back to global config
    self.proxy = bool(self.config.get("proxy", _model_config.get("proxy", False)));
    self.http_client = bool(
        self.config.get("http_client", _model_config.get("http_client", False))
    );
    # Resolve api_key: constructor > instance config > global config > env vars (litellm fallback)
    self.api_key = (
        self.api_key
        or self.config.get("api_key", "")
        or _model_config.get("api_key", "")
    );
    # When model_name is 'mockllm', delegate to MockLLM behavior
    if self.model_name == MODEL_MOCK {
        self._mock_delegate = MockLLM(model_name=self.model_name, config=self.config);
    }
    if not self._mock_delegate {
        logging.getLogger("httpx").setLevel(logging.WARNING);
        _disable_debugging();
        # Use drop_params from config
        litellm.drop_params = _litellm_config.get("drop_params", True);
    }
}

"""Invoke the LLM, delegating to MockLLM if applicable.

Wraps the full invocation (including ReAct loop) with telemetry.
Generates a unique invocation_id shared across all LLM calls in
this agent invocation via make_model_params metadata.
"""
impl Model.invoke(mt_run: MTRuntime) -> object {
    import uuid;
    import time;

    # Generate a unique ID for this entire agent invocation
    # (shared across all LLM calls in the ReAct loop via make_model_params metadata)
    mt_run._invocation_id = str(uuid.uuid4());
    _start = time.time();

    # Extract telemetry context before the LLM call — available for both success and error paths
    user_prompt = "";
    try {
        user_prompt = str(mt_run.messages[1].content[0].text)[:2000];
    } except Exception as e {
        logging.getLogger("byllm.telemetry").debug(
            f"user_prompt extraction failed: {e}"
        );
    }
    caller_args: dict = {};
    try {
        if mt_run.mtir and mt_run.mtir.args {
            for k in mt_run.mtir.args {
                caller_args[str(k)] = str(mt_run.mtir.args[k])[:500];
            }
        }
    } except Exception as e {
        logging.getLogger("byllm.telemetry").debug(
            f"caller_args extraction failed: {e}"
        );
    }
    conv_history: list = [];
    try {
        conv_history = mt_run.get_msg_list();
    } except Exception as e {
        logging.getLogger("byllm.telemetry").debug(
            f"conv_history extraction failed: {e}"
        );
    }

    try {
        # Delegate to mock or real LLM — existing logic, untouched
        if self._mock_delegate {
            result = self._mock_delegate.invoke(mt_run);
        } else {
            result = super.invoke(mt_run);
        }

        # Emit agent-level telemetry on success (fire-and-forget)
        try {
            import from byllm.telemetry { _emit_agent_telemetry }
            # Extract the real agent response text
            import types as _types_mod;
            _agent_resp = "";
            if isinstance(result, _types_mod.GeneratorType) {
                # Streaming: response not yet generated at this point
                _agent_resp = "[streaming]";
            } else {
                # Non-streaming: use the parsed result directly
                _agent_resp = str(result)[:2000];
            }
            # Re-capture conversation history after LLM call (may have new messages)
            try {
                conv_history = mt_run.get_msg_list();
            } except Exception as e {
                logging.getLogger("byllm.telemetry").debug(
                    f"post-call conv_history refresh failed: {e}"
                );
            }
            _emit_agent_telemetry(
                {
                    "invocation_id": mt_run._invocation_id,
                    "caller_name": mt_run.mtir.caller.__name__
                        if mt_run.mtir
                        else "unknown",
                    "caller_args": caller_args,
                    "user_prompt": user_prompt,
                    "agent_response": _agent_resp,
                    "conversation_history": conv_history,
                    "model": self.model_name,
                    "latency_ms": (time.time() - _start) * 1000,
                    "status": "success",
                    "error": None
                }
            );
        } except Exception as _te {
            logging.getLogger("byllm.telemetry").warning(
                f"Agent telemetry emit failed: {_te}"
            );
        }

        return result;
    } except Exception as e {
        # Emit failure telemetry
        try {
            import from byllm.telemetry { _emit_agent_telemetry }
            _emit_agent_telemetry(
                {
                    "invocation_id": mt_run._invocation_id,
                    "caller_name": mt_run.mtir.caller.__name__
                        if mt_run.mtir
                        else "unknown",
                    "caller_args": caller_args,
                    "user_prompt": user_prompt,
                    "agent_response": "",
                    "conversation_history": conv_history,
                    "model": self.model_name,
                    "latency_ms": (time.time() - _start) * 1000,
                    "status": "error",
                    "error": str(e)[:1000]
                }
            );
        } except Exception as e {
            logging.getLogger("byllm.telemetry").debug(
                f"Error-path telemetry emit failed: {e}"
            );
        }
        raise;
    }
}

"""Make a direct model call without streaming.

Uses OpenAI client if proxy mode is enabled, otherwise uses LiteLLM.
If _http_client is provided in config, uses direct HTTP requests.
"""
impl Model.model_call_no_stream(params: dict) -> dict {
    try {
        if self.http_client {
            response = self.model_call_http(
                url=self.api_base,
                token=self.api_key,
                payload=params,
                ca_bundle=self.config.get("ca_bundle", None),
            );
        } elif self.proxy {
            client = OpenAI(base_url=self.api_base, api_key=self.api_key or None);
            response = client.chat.completions.create(**params);
        } else {
            response = litellm.completion(api_key=self.api_key or None, **params);
        }
    } except litellm.exceptions.AuthenticationError as e {
        raise AuthenticationError(str(e));
    } except openai.AuthenticationError as e {
        raise AuthenticationError(str(e));
    } except litellm.exceptions.RateLimitError as e {
        raise RateLimitError(str(e));
    } except openai.RateLimitError as e {
        raise RateLimitError(str(e));
    } except litellm.exceptions.NotFoundError as e {
        raise ModelNotFoundError(str(e));
    } except openai.NotFoundError as e {
        raise ModelNotFoundError(str(e));
    }
    return response;
}

"""Make a direct model call with streaming.

Uses OpenAI client if proxy mode is enabled, otherwise uses LiteLLM.
Returns a generator that yields response chunks.
"""
impl Model.model_call_with_stream(params: dict) {
    try {
        if self.proxy {
            client = OpenAI(base_url=self.api_base, api_key=self.api_key or None);
            response = client.chat.completions.create(stream=True, **params);
        } else {
            response = litellm.completion(
                stream=True, api_key=self.api_key or None, **params
            );
        }
    } except litellm.exceptions.AuthenticationError as e {
        raise AuthenticationError(str(e));
    } except openai.AuthenticationError as e {
        raise AuthenticationError(str(e));
    } except litellm.exceptions.RateLimitError as e {
        raise RateLimitError(str(e));
    } except openai.RateLimitError as e {
        raise RateLimitError(str(e));
    } except litellm.exceptions.NotFoundError as e {
        raise ModelNotFoundError(str(e));
    } except openai.NotFoundError as e {
        raise ModelNotFoundError(str(e));
    }
    return response;
}

"""Make a direct HTTP request to LLM endpoint.

Useful for custom endpoints with specific SSL/mTLS requirements.
"""
impl Model.model_call_http(
    url: str, token: str, payload: dict, ca_bundle: str
) -> object {
    import requests;
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"};
    response = requests.post(
        url=url, headers=headers, json=payload, timeout=60, verify=ca_bundle or True,
    );
    return response.json();
}
