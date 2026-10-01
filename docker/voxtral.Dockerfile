# vLLM + the audio extras needed by /v1/audio/transcriptions (the stock image
# lacks the decoders), used by the dreamer's on-demand Voxtral ASR service.
FROM vllm/vllm-openai:v0.27.1
RUN pip install --no-cache-dir "av" "soundfile" "mistral_common[soundfile]"
