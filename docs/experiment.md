# The experiment

Are LLMs good at faking human behaviour by just letting them write inside the user slot of the chat template?

Every model is tested in two ways: **native**, writing in the user slot like selfgpt does, and **roleplay**, which just asks the model to play the human with a prompt.

```bash
# Get the data: ShareChat is gated, request access on Hugging Face then put chatgpt_results_final_language_filtered.csv in data/sharechat/

# Download the models into models/
./download.sh

# Start llama-server (brew install llama.cpp) and leave it running
llama-server --models-dir models --models-preset eval/llama-server.ini --models-max 1 -c 10240 -np 32 -kvu -fa auto -cram 256 -ctxcp 0 --port 9000

# Run evaluation pipeline
JUDGE_API_KEY=$OPENROUTER_API_KEY python -m eval.run models/*.gguf --judge-url https://openrouter.ai/api/v1 --judge-model deepseek/deepseek-v4.1-flash-1017 --judge-extra '{"provider": {"require_parameters": true, "sort": "price"}}'
```

## Judges

| Flag | Effect |
|---|---|
| `--judge-provider api` | Any OpenAI-compatible API that returns logprobs (default) |
| `--judge-provider openai-oauth` | Use a ChatGPT Plus/Pro plan, log in with `python -m eval.chatgpt login` |
| `--no-logprobs` | For APIs without logprobs |

## Results

Everything ends up in `eval/out/`: `results.md` has every result and the judge checks, `samples.md` every generated turn next to the real one. Rebuild the report alone with `python -m eval.report eval/out`.
