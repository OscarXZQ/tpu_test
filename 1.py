from vllm import LLM, SamplingParams

def sample_model(prompt: str, num_samples: int = 3, temperature: float = 0.8, top_p: float = 0.9):
    """
    Generate multiple completions from Qwen-4B using vLLM on TPU.
    """
    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=200,
        n=num_samples
    )

    # Initialize the LLM (vLLM will automatically use the TPU setup)
    llm = LLM(model="Qwen/Qwen3-4B")

    # Run generation
    outputs = llm.generate(prompt, sampling_params)

    # Print results
    print(f"\nPrompt: {prompt}\n{'='*60}")
    for i, output in enumerate(outputs, start=1):
        print(f"[Sample {i}] {output.outputs[0].text.strip()}\n{'-'*60}")

if __name__ == "__main__":
    # Example prompt
    test_prompt = "Write a short poem about dawn in the mountains."
    sample_model(test_prompt, num_samples=3)
