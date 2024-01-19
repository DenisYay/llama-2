# Distributed inference and serving API server example of Meta LLaMa-2-13B using Runhouse (https://run.house) 
# and vLLM (https://docs.vllm.ai).
#
# To run: python rh_llama2_vllm_module.py

import runhouse as rh
from typing import AsyncGenerator, Generator, Optional, Any
import asyncio
import json

class Llama2Model(rh.Module):
    def __init__(self, model_id="openlm-research/open_llama_13b"):
        super().__init__()
        self.model_id = model_id
        self.engine = None
    
    def initialize_engine(self) -> None:
        """Initialize the AsyncLLMEngine from the command line arguments."""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        
        print(f"\n\n... Initializing engine ...\n")
        args = AsyncEngineArgs(model=self.model_id, tensor_parallel_size=4, trust_remote_code=True, enforce_eager=True)
        engine_args = AsyncEngineArgs.from_cli_args(args)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self, prompt: str, stream: bool = True, sampling_params: Optional[dict] = None) -> Any:
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid
        from typing import AsyncGenerator
        import json

        """Generate completion for the request.
        - prompt: the prompt to use for the generation.
        - stream: whether to stream the results or not.
        - other fields: the sampling parameters (See `SamplingParams` for details).
        """
        if isinstance(sampling_params, dict):
          sampling_params = SamplingParams(**sampling_params)  # Unpack dictionary
        elif isinstance(sampling_params, str):
          sampling_params = SamplingParams.from_json(sampling_params)  # Parse JSON string
        else:
          raise ValueError("Invalid type for sampling_params")
        request_id = random_uuid()

        if self.engine is None:
          self.initialize_engine()

        results_generator = self.engine.generate(prompt, sampling_params, request_id)
    
        async def stream_results() -> AsyncGenerator[bytes, None]:
          try:
              print("\n\nBefore going to async for loop\n")

              async for request_output in results_generator:
                  print(f"Debug: request_output = {request_output}")

                  prompt = request_output.prompt
                  text_outputs = [
                      prompt + output.text for output in request_output.outputs
                  ]
                  print(f"... Async Generated Text:\n{text_outputs}\n")

                  ret = {"text": text_outputs}
                  yield (json.dumps(ret) + "\0").encode("utf-8")

          except Exception as e:
            print(f"Error in stream_results: {e}")
            raise
        
        # Streaming case
        if stream:
            return stream_results()

        # Non-streaming case
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        assert final_output is not None
        prompt = final_output.prompt
        text_outputs = [prompt + output.text for output in final_output.outputs]
        return text_outputs

async def main():
    env = rh.env(reqs=["vllm"])
    gpu = rh.ondemand_cluster(name='rh-a10x', env=env, instance_type='A10G:4').up_if_not()
    print(f"\n\n... Initializing LLaMa-2 model ...\n")
    remote_llama2_model = Llama2Model().get_or_to(system=gpu, env=env, name="llama-2-model")
    print(f"\n\n... Succefully initialized LLaMa-2 model. Running generation ...\n")
    ans =  await remote_llama2_model.generate(prompt="Wheels on the bus go", stream=False, sampling_params={"temperature": 0.8, "top_p": 0.95})
    for text_output in ans:
        print(f"... Generated Text:\n{text_output}\n")


if __name__ == "__main__":
    asyncio.run(main())
