"""byLLM Package."""
import from byllm.llm { MockLLM, Model, ModelPool }
import from byllm.mtir { MTIR, MTRuntime }
import from byllm.plugin { JacRuntime }
import from byllm.types { Image, MockToolCall, StreamEvent, Video }
import from byllm.exceptions {
    ByLLMError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    OutputConversionError,
    UnknownToolError,
    FinishToolError,
    ConfigurationError
}
import from byllm.telemetry { register_agent_callback }
import from byllm.parallel { dispatch_batch, mark_serialize }

glob by = JacRuntime.by ,
     __all__ = [
         'by',
         'ByLLMError',
         'AuthenticationError',
         'RateLimitError',
         'ModelNotFoundError',
         'OutputConversionError',
         'UnknownToolError',
         'FinishToolError',
         'ConfigurationError',
         'Image',
         'MockLLM',
         'MockToolCall',
         'Model',
         'ModelPool',
         'MTIR',
         'MTRuntime',
         'StreamEvent',
         'Video',
         'register_agent_callback',
         'dispatch_batch',
         'mark_serialize'
     ];
