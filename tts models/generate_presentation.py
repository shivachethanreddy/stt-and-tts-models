import os
import json
import numpy as np
import soundfile as sf
import onnxruntime as ort

text = """When disaster strikes, communication infrastructure often fails first. Cell towers go down. Internet disappears. But information still needs to reach everyone — not just those who can read.

This is iTantra. An offline, multilingual voice communication system, built for exactly that moment.

Two phones talk like a walkie talkie. Entirely offline. In ten Indian languages. No internet. No SIM. No cellular network.

One phone listens, and converts speech into text, locally. That text travels to another phone over a direct wireless link. The receiving phone converts it back into speech, and speaks it out loud.

We never send raw audio. Every sentence becomes a tiny text packet — around one hundred twenty bytes. Over ninety nine percent smaller than the same message as audio. That's what makes this work on a low bandwidth link.

Let's see it in action.

Two phones, connected directly. No router. No internet. I'll press and hold to talk.

Now, the emergency case. An alert message plays at maximum volume, and cannot be interrupted — because in a real distress situation, it has to get through.

Underneath this: on-device A I for speech, in ten languages, running on ordinary low and mid range phones. Wi-Fi Direct as the primary link, with Bluetooth as automatic backup. And for alerts specifically, a mesh relay that lets a message hop phone to phone, reaching everyone nearby — not just one paired device.

iTantra. Voice that reaches everyone, even when everything else has gone silent.

Built by team Monte Carlo. Smart India Hackathon, twenty twenty six. Problem statement, two six one seven three."""

def tokenize_mms(text, vocab):
    cleaned = [c for c in text.lower() if c in vocab]
    token_ids = [vocab[c] for c in cleaned]
    interspersed = [0] * (len(token_ids) * 2 + 1)
    interspersed[1::2] = token_ids
    return interspersed, len(cleaned)

folder = "english"
onnx_file = os.path.join(folder, "model_optimized.onnx")
vocab_file = os.path.join(folder, "vocab.json")

with open(vocab_file, encoding='utf-8') as f:
    vocab = json.load(f)

token_ids, matched_char_count = tokenize_mms(text, vocab)

session = ort.InferenceSession(onnx_file)
inp = np.array([token_ids], dtype=np.int64)
mask = np.ones_like(inp, dtype=np.int64)

outputs = session.run(None, {"input_ids": inp, "attention_mask": mask})
waveform = outputs[0][0]

out_path = "generated_audio/presentation.wav"
os.makedirs("generated_audio", exist_ok=True)
sf.write(out_path, waveform, 16000)

print(f"Audio generated successfully at {os.path.abspath(out_path)}")
