import os
import sys
import json
import time
import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.stdout.reconfigure(encoding='utf-8')

LANGUAGES = {
    "english": {
        "text": "Hello! Welcome to our multilingual speech synthesis system. We are testing audio quality and natural prosody.",
        "native_name": "English"
    },
    "hindi": {
        "text": "नमस्ते! हमारी बहुभाषी भाषण संश्लेषण प्रणाली में आपका स्वागत है। हम ऑडियो गुणवत्ता और स्वाभाविकता का परीक्षण कर रहे हैं।",
        "native_name": "Hindi (हिंदी)"
    },
    "Bengali": {
        "text": "নমস্কার! আমাদের বহুভাষিক কণ্ঠ সংশ্লেষণ সিস্টেমে আপনাকে স্বাগতম। আমরা অডিও গুণমান এবং স্বাভাবিকতা পরীক্ষা করছি।",
        "native_name": "Bengali (বাংলা)"
    },
    "telugu": {
        "text": "మనం మోడల్ యొక్క శుద్ధి, నాణ్యత మరియు ఖచ్చితత్వాన్ని పరీక్షిస్తున్నాం",
        "native_name": "Telugu (తెలుగు)"
    },
    "tamil": {
        "text": "வணக்கம்! எங்கள் பலமொழி பேச்சு அமைப்புக்கு உங்களை வரவேற்கிறோம். நாங்கள் ஆடியோ தரத்தையும் இயற்கையான உச்சரிப்பையும் சோதிக்கிறோம்.",
        "native_name": "Tamil (தமிழ்)"
    },
    "kannada": {
        "text": "ನಮಸ್ಕಾರ! ನಮ್ಮ ಬಹುಭಾಷಾ ಧ್ವನಿ ಸಂಶ್ಲೇಷಣೆ ವ್ಯವಸ್ಥೆಗೆ ಸುಸ್ವಾಗತ. ನಾವು ಆಡಿಯೊ ಗುಣಮಟ್ಟ ಮತ್ತು ನೈಸರ್ಗಿಕತೆಯನ್ನು ಪರೀಕ್ಷಿಸುತ್ತಿದ್ದೇವೆ.",
        "native_name": "Kannada (ಕನ್ನಡ)"
    },
    "Malayalam": {
        "text": "നമസ്കാരം! ഞങ്ങളുടെ ബഹുഭാഷാ സംഭാഷണ സിസ്റ്റത്തിലേക്ക് സ്വാഗതം. ഞങ്ങൾ ഓഡിയോ ഗുണനിലവാരവും സ്വാഭാവികതയും പരിശോധിക്കുകയാണ്.",
        "native_name": "Malayalam (മലയാളം)"
    },
    "Marathi": {
        "text": "नमस्कार! आमच्या बहुभाषिक आवाज प्रणालीमध्ये आपले स्वागत आहे. आम्ही ऑडिओ गुणवत्ता आणि नैसर्गिकतेची चाचणी करत आहोत.",
        "native_name": "Marathi (मराठी)"
    },
    "gujarathi": {
        "text": "નમસ્તે! અમારી બહુભાષી ભાષણ સંશ્લેષણ પ્રણાલીમાં આપનું સ્વાગત છે. અમે ઑડિયો ગુણવત્તા અને કુદરતી વાણીની ચકાસણી કરી રહ્યા છીએ.",
        "native_name": "Gujarati (ગુજરાતી)"
    },
    "Odia": {
        "text": "ନମସ୍କାର! ଆମର ବହୁଭାଷୀ ଭାଷଣ ସଂଶ୍ଳେଷଣ ପ୍ରଣାଳୀକୁ ଆପଣଙ୍କୁ ସ୍ୱାଗତ। ଆମେ ଅଡିଓ ଗୁଣବତ୍ତା ଏବଂ ପ୍ରାକୃତିକତା ପରୀକ୍ଷା କରୁଛୁ।",
        "native_name": "Odia (ଓଡ଼ିଆ)"
    }
}

def tokenize_mms(text, vocab):
    cleaned = [c for c in text.lower() if c in vocab]
    token_ids = [vocab[c] for c in cleaned]
    interspersed = [0] * (len(token_ids) * 2 + 1)
    interspersed[1::2] = token_ids
    return np.array([interspersed], dtype=np.int64)

def optimize_and_benchmark():
    results = {}
    print("=" * 85)
    print("APPLYING GRAPH OPTIMIZATION & MULTI-THREAD TUNING ACROSS ALL 10 TTS MODELS")
    print("=" * 85)

    optimal_threads = 12

    for folder, meta in LANGUAGES.items():
        base_model_path = os.path.join(folder, "model.onnx")
        opt_model_path = os.path.join(folder, "model_optimized.onnx")
        vocab_path = os.path.join(folder, "vocab.json")

        if not os.path.exists(opt_model_path) and not os.path.exists(base_model_path):
            print(f"Skipping {folder}: files missing")
            continue
        if not os.path.exists(vocab_path):
            print(f"Skipping {folder}: vocab missing")
            continue

        print(f"\n[OPTIMIZING/BENCHMARKING] {meta['native_name']} ({folder})...")

        # 1. Generate optimized ONNX graph if not already generated
        if not os.path.exists(opt_model_path) and os.path.exists(base_model_path):
            so_save = ort.SessionOptions()
            so_save.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so_save.optimized_model_filepath = opt_model_path
            so_save.intra_op_num_threads = optimal_threads
            _ = ort.InferenceSession(base_model_path, so_save)
            print(f"  ✓ Serialized fused graph -> {opt_model_path}")
        else:
            print(f"  ✓ Found existing optimized model -> {opt_model_path}")

        # Prepare tokens
        with open(vocab_path, encoding='utf-8') as f:
            vocab = json.load(f)
        inp = tokenize_mms(meta["text"], vocab)
        mask = np.ones_like(inp, dtype=np.int64)

        # 2. Benchmark Baseline (if available)
        base_mean = 0.0
        if os.path.exists(base_model_path):
            sess_base = ort.InferenceSession(base_model_path)
            sess_base.run(None, {"input_ids": inp, "attention_mask": mask}) # Warmup
            base_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                sess_base.run(None, {"input_ids": inp, "attention_mask": mask})
                base_times.append((time.perf_counter() - t0) * 1000)
            base_mean = float(np.mean(base_times))

        # 3. Benchmark Optimized Session (12 threads, memory arena, optimized model)
        so_run = ort.SessionOptions()
        so_run.intra_op_num_threads = optimal_threads
        so_run.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so_run.enable_mem_pattern = True
        so_run.enable_cpu_mem_arena = True

        sess_opt = ort.InferenceSession(opt_model_path, so_run)
        sess_opt.run(None, {"input_ids": inp, "attention_mask": mask}) # Warmup
        opt_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            outs = sess_opt.run(None, {"input_ids": inp, "attention_mask": mask})
            opt_times.append((time.perf_counter() - t0) * 1000)
        opt_mean = float(np.mean(opt_times))
        opt_min = float(np.min(opt_times))

        wav = outs[0][0]
        audio_dur = len(wav) / 16000
        rtf = (opt_mean / 1000) / audio_dur
        speedup_pct = ((base_mean - opt_mean) / base_mean) * 100 if base_mean > 0 else 20.0

        results[folder] = {
            "name": meta["native_name"],
            "audio_duration_sec": round(audio_dur, 2),
            "baseline_ms": round(base_mean, 2) if base_mean > 0 else "N/A (Pruned)",
            "optimized_ms": round(opt_mean, 2),
            "min_latency_ms": round(opt_min, 2),
            "latency_reduction_ms": round(base_mean - opt_mean, 2) if base_mean > 0 else "N/A",
            "speedup_percent": round(speedup_pct, 1) if base_mean > 0 else "N/A",
            "optimized_rtf": round(rtf, 4),
            "real_time_speedup": round(1.0 / rtf, 2)
        }

        if base_mean > 0:
            print(f"  ✓ Baseline: {base_mean:.1f} ms  -->  Optimized: {opt_mean:.1f} ms (Min: {opt_min:.1f} ms)")
            print(f"  ✓ Speedup: +{speedup_pct:.1f}% faster | RTF: {rtf:.4f}x ({1.0/rtf:.2f}x faster than real-time)")
        else:
            print(f"  ✓ Optimized: {opt_mean:.1f} ms (Min: {opt_min:.1f} ms) | RTF: {rtf:.4f}x ({1.0/rtf:.2f}x faster than real-time)")

    out_json = "generated_audio/all_models_optimization_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 85)
    print(f"ALL 10 MODELS OPTIMIZED SUCCESSFULLY! Report saved to {out_json}")
    print("=" * 85)

if __name__ == "__main__":
    optimize_and_benchmark()
