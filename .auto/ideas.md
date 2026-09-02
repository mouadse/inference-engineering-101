# Optimization ideas

- If dense Qwen3.5 support lands in a usable TensorRT-LLM image during the session, test the PyTorch backend with CUDA graphs and C++ sampling before trying AutoDeploy compilation.
- Test safe weight-only INT8/INT4 only if an exact Qwen3.5 checkpoint/runtime path is available and add a quality check first; T4 lacks native FP8/BF16 acceleration.
- Add server-side decode profiling only if simple launch/configuration experiments plateau; client HTTP overhead is expected to be small at 128–256 output tokens.
