import os
import json
import numpy as np
import soundfile as sf
import onnxruntime as ort

text = """जब कोई आपदा आती है, तो अक्सर सबसे पहले संचार व्यवस्था ही ठप हो जाती है। मोबाइल टावर बंद हो जाते हैं। इंटरनेट गायब हो जाता है। लेकिन जानकारी फिर भी हर व्यक्ति तक पहुँचनी चाहिए — सिर्फ़ उन लोगों तक नहीं जो पढ़ सकते हैं।

यही है iTantra — एक ऑफलाइन, बहुभाषी वॉइस कम्युनिकेशन सिस्टम, जिसे बिल्कुल ऐसी ही आपातकालीन परिस्थितियों के लिए बनाया गया है।

दो फोन आपस में वॉकी-टॉकी की तरह बात करते हैं। पूरी तरह ऑफलाइन। भारत की दस भाषाओं में। न इंटरनेट की जरूरत, न SIM की, और न ही किसी मोबाइल नेटवर्क की।

एक फोन आपकी आवाज़ सुनता है और उसे फोन के अंदर ही टेक्स्ट में बदल देता है। यह टेक्स्ट सीधे वायरलेस लिंक के जरिए दूसरे फोन तक पहुँचता है। रिसीव करने वाला फोन उस टेक्स्ट को दोबारा आवाज़ में बदलकर ज़ोर से सुनाता है।

हम कभी भी कच्ची ऑडियो फाइल नहीं भेजते। हर वाक्य लगभग 120 बाइट्स के एक छोटे टेक्स्ट पैकेट में बदल जाता है। उसी संदेश को ऑडियो के रूप में भेजने की तुलना में यह 99 प्रतिशत से भी ज्यादा छोटा है। यही वजह है कि यह कम बैंडविड्थ वाले लिंक पर भी काम कर सकता है।

आइए, इसे काम करते हुए देखते हैं।

दो फोन सीधे एक-दूसरे से जुड़े हुए हैं। कोई राउटर नहीं। कोई इंटरनेट नहीं। मैं बोलने के लिए बटन को दबाकर रखूँगा।

अब देखते हैं आपातकालीन स्थिति।

एक अलर्ट मैसेज अधिकतम वॉल्यूम पर चलेगा और उसे बीच में रोका नहीं जा सकता — क्योंकि वास्तविक आपदा की स्थिति में जरूरी है कि अलर्ट हर हाल में लोगों तक पहुँचे।

इसके पीछे है ऑन-डिवाइस AI, जो दस भाषाओं में स्पीच को प्रोसेस करता है और सामान्य लो-एंड और मिड-रेंज फोन पर ही चलता है।

मुख्य कम्युनिकेशन लिंक के रूप में Wi-Fi Direct का इस्तेमाल किया जाता है, जबकि Bluetooth ऑटोमैटिक बैकअप के रूप में काम करता है।

और खास तौर पर अलर्ट के लिए, इसमें Mesh Relay की सुविधा भी है। इसके जरिए एक संदेश फोन से फोन तक आगे बढ़ता है और आसपास मौजूद सभी लोगों तक पहुँच सकता है — सिर्फ़ एक पेयर्ड डिवाइस तक सीमित नहीं रहता।

iTantra.

एक ऐसी आवाज़, जो हर किसी तक पहुँचे — तब भी, जब बाकी सब कुछ खामोश हो जाए।

Team Monte Carlo द्वारा निर्मित।

Smart India Hackathon 2026.

Problem Statement: 26173."""

def tokenize_mms(text, vocab):
    cleaned = [c for c in text.lower() if c in vocab]
    token_ids = [vocab[c] for c in cleaned]
    interspersed = [0] * (len(token_ids) * 2 + 1)
    interspersed[1::2] = token_ids
    return interspersed, len(cleaned)

folder = "hindi"
onnx_file = os.path.join(folder, "model_optimized.onnx")
vocab_file = os.path.join(folder, "vocab.json")

if not os.path.exists(onnx_file):
    onnx_file = os.path.join(folder, "model.onnx")

with open(vocab_file, encoding='utf-8') as f:
    vocab = json.load(f)

token_ids, matched_char_count = tokenize_mms(text, vocab)

session = ort.InferenceSession(onnx_file)
inp = np.array([token_ids], dtype=np.int64)
mask = np.ones_like(inp, dtype=np.int64)

outputs = session.run(None, {"input_ids": inp, "attention_mask": mask})
waveform = outputs[0][0]

out_path = "generated_audio/presentation_hindi.wav"
os.makedirs("generated_audio", exist_ok=True)
sf.write(out_path, waveform, 16000)

print(f"Audio generated successfully at {os.path.abspath(out_path)}")
