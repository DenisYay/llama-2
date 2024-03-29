# Distributed inference and serving API server example of Meta LLaMa-2-13B using Runhouse (https://run.house) 
# and vLLM (https://docs.vllm.ai).
#
# To run: python rh_llama2_vllm_module.py

import runhouse as rh
from typing import Optional, Any
import asyncio

class Llama2Model(rh.Module):
    def __init__(self, model_id="openlm-research/open_llama_13b"):
        super().__init__()
        self.model_id = model_id
        self.engine = None
    
    def initialize_engine(self) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        
        print(f"\n\n... Initializing engine ...\n")
        args = AsyncEngineArgs(model=self.model_id, tensor_parallel_size=4, trust_remote_code=True, enforce_eager=True)
        engine_args = AsyncEngineArgs.from_cli_args(args)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def generate(self, prompt: str, stream: bool = True, **sampling_params: Optional[dict]) -> Any:
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid
        from typing import AsyncGenerator

        sampling_params = SamplingParams(**sampling_params)  # Unpack dictionary
        request_id = random_uuid()

        if not self.engine:
          self.initialize_engine()

        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        if stream:
            async for request_output in results_generator:
                prompt = request_output.prompt
                text_outputs = [
                    prompt + output.text for output in request_output.outputs
                ]
                yield text_outputs
    
async def main():
    env = rh.env(reqs=["vllm"])
    gpu = rh.ondemand_cluster(name='rh-a10x', env=env, instance_type='A10G:4').up_if_not()
    print(f"\n\n... Initializing LLaMa-2 model ...\n")
    remote_llama2_model = Llama2Model().to(system=gpu, env=env, name="llama-2-model")
    print(f"\n\n... Succefully initialized LLaMa-2 model. Running generation ...\n")

    stream = True
    ans = remote_llama2_model.generate(prompt="Wheels on the bus go", stream=stream, temperature=0.8, top_p=0.95, max_tokens=200)
    print(f"\n\n... type of ans = {type(ans)}\n")
    async for text_output in ans:
        print(f"\n\n... Generated Text:\n{text_output}\n")


if __name__ == "__main__":
    asyncio.run(main())
