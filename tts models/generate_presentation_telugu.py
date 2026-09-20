import os
import json
import numpy as np
import soundfile as sf
import onnxruntime as ort

text = """విపత్తు సంభవించినప్పుడు, కమ్యూనికేషన్ వ్యవస్థలే తరచుగా ముందుగా పనిచేయడం ఆపేస్తాయి. సెల్ టవర్లు పనిచేయడం ఆగిపోతాయి. ఇంటర్నెట్ కనెక్షన్ ఉండదు. కానీ సమాచారం మాత్రం ప్రతి ఒక్కరికీ చేరాలి — చదవగలిగిన వారికి మాత్రమే కాదు.

ఇదే iTantra. ఆఫ్లైన్లో పనిచేసే, బహుభాషా వాయిస్ కమ్యూనికేషన్ సిస్టమ్. సరిగ్గా ఇలాంటి అత్యవసర పరిస్థితుల కోసం రూపొందించబడింది.

రెండు ఫోన్లు వాకీ-టాకీలా ఒకదానితో ఒకటి మాట్లాడుకుంటాయి. పూర్తిగా ఆఫ్లైన్లో. పది భారతీయ భాషల్లో. ఇంటర్నెట్ అవసరం లేదు. సిమ్ అవసరం లేదు. సెల్యులార్ నెట్వర్క్ అవసరం లేదు.

ఒక ఫోన్ మన మాటలను వింటుంది, వాటిని ఫోన్లోనే టెక్స్ట్గా మారుస్తుంది. ఆ టెక్స్ట్ నేరుగా వైర్లెస్ లింక్ ద్వారా మరో ఫోన్కు చేరుతుంది. రిసీవ్ చేసే ఫోన్ ఆ టెక్స్ట్ను మళ్లీ వాయిస్గా మార్చి, గట్టిగా వినిపిస్తుంది.

మేము ఎప్పుడూ అసలు ఆడియోను పంపించము. ప్రతి వాక్యం దాదాపు 120 బైట్ల చిన్న టెక్స్ట్ ప్యాకెట్గా మారుతుంది. అదే సందేశాన్ని ఆడియోగా పంపడంతో పోలిస్తే ఇది 99 శాతానికి పైగా చిన్నది. అందుకే తక్కువ బ్యాండ్విడ్త్ ఉన్న లింక్పైనా ఇది పనిచేస్తుంది.

ఇప్పుడు ఇది ఎలా పనిచేస్తుందో చూద్దాం.

రెండు ఫోన్లు నేరుగా ఒకదానితో ఒకటి కనెక్ట్ అయ్యాయి. రూటర్ లేదు. ఇంటర్నెట్ లేదు. మాట్లాడేందుకు నేను బటన్ను నొక్కి పట్టుకుంటాను.

ఇప్పుడు అత్యవసర పరిస్థితిని చూద్దాం. ఒక అలర్ట్ మెసేజ్ గరిష్ఠ వాల్యూమ్లో ప్లే అవుతుంది. దాన్ని మధ్యలో ఆపలేరు — ఎందుకంటే నిజమైన విపత్తు పరిస్థితుల్లో ఆ సందేశం తప్పనిసరిగా చేరాలి.

దీని వెనుక ఉన్న సాంకేతికత: పది భాషల్లో స్పీచ్ను అర్థం చేసుకునే ఆన్-డివైస్ AI, ఇది సాధారణ లో-ఎండ్ మరియు మిడ్-రేంజ్ ఫోన్లపైనే పనిచేస్తుంది.

ప్రధాన కమ్యూనికేషన్ లింక్గా Wi-Fi Direct, అది అందుబాటులో లేకపోతే ఆటోమేటిక్ బ్యాకప్గా Bluetooth ఉపయోగించబడుతుంది.

ముఖ్యంగా అలర్ట్ల కోసం, మెష్ రిలే సిస్టమ్ కూడా ఉంది. దీని ద్వారా ఒక మెసేజ్ ఫోన్ నుంచి ఫోన్కు హాప్ అవుతూ, సమీపంలోని ప్రతి ఒక్కరికీ చేరుతుంది — కేవలం ఒక జత చేసిన డివైస్కు మాత్రమే కాదు.

iTantra.

మిగతావన్నీ నిశ్శబ్దమైనా, ప్రతి ఒక్కరికీ చేరే వాయిస్.

Team Monte Carlo రూపొందించింది.

Smart India Hackathon 2026.

Problem Statement: 26173."""

def tokenize_mms(text, vocab):
    cleaned = [c for c in text.lower() if c in vocab]
    token_ids = [vocab[c] for c in cleaned]
    interspersed = [0] * (len(token_ids) * 2 + 1)
    interspersed[1::2] = token_ids
    return interspersed, len(cleaned)

folder = "telugu"
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

out_path = "generated_audio/presentation_telugu.wav"
os.makedirs("generated_audio", exist_ok=True)
sf.write(out_path, waveform, 16000)

print(f"Audio generated successfully at {os.path.abspath(out_path)}")
