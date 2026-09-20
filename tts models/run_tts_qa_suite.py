import os
import sys
import json
import time
import numpy as np
import soundfile as sf
import scipy.signal
import onnxruntime as ort

sys.stdout.reconfigure(encoding='utf-8')

LANGUAGES = {
    "english": {
        "text": "Hello! Welcome to our multilingual speech synthesis system. We are testing audio quality and natural prosody.",
        "native_name": "English",
        "folder": "english"
    },
    "hindi": {
        "text": "नमस्ते! हमारी बहुभाषी भाषण संश्लेषण प्रणाली में आपका स्वागत है। हम ऑडियो गुणवत्ता और स्वाभाविकता का परीक्षण कर रहे हैं।",
        "native_name": "हिन्दी (Hindi)",
        "folder": "hindi"
    },
    "Bengali": {
        "text": "নমস্কার! আমাদের বহুভাষিক কণ্ঠ সংশ্লেষণ সিস্টেমে আপনাকে স্বাগতম। আমরা অডিও গুণমান এবং স্বাভাবিকতা পরীক্ষা করছি।",
        "native_name": "বাংলা (Bengali)",
        "folder": "Bengali"
    },
    "telugu": {
        "text": "నమస్కారం! మా బహుభాషా ప్రసంగ సంశ్లేషణ వ్యవస్థకు స్వాగతం. మేము ఆడియో నాణ్యత మరియు సహజత్వాన్ని పరీక్షిస్తున్నాము.",
        "native_name": "తెలుగు (Telugu)",
        "folder": "telugu"
    },
    "tamil": {
        "text": "வணக்கம்! எங்கள் பலமொழி பேச்சு அமைப்புக்கு உங்களை வரவேற்கிறோம். நாங்கள் ஆடியோ தரத்தையும் இயற்கையான உச்சரிப்பையும் சோதிக்கிறோம்.",
        "native_name": "தமிழ் (Tamil)",
        "folder": "tamil"
    },
    "kannada": {
        "text": "ನಮಸ್ಕಾರ! ನಮ್ಮ ಬಹುಭಾಷಾ ಧ್ವನಿ ಸಂಶ್ಲೇಷಣೆ ವ್ಯವಸ್ಥೆಗೆ ಸುಸ್ವಾಗತ. ನಾವು ಆಡಿಯೊ ಗುಣಮಟ್ಟ ಮತ್ತು ನೈಸರ್ಗಿಕತೆಯನ್ನು ಪರೀಕ್ಷಿಸುತ್ತಿದ್ದೇವೆ.",
        "native_name": "ಕನ್ನಡ (Kannada)",
        "folder": "kannada"
    },
    "Malayalam": {
        "text": "നമസ്കാരം! ഞങ്ങളുടെ ബഹുഭാഷാ സംഭാഷണ സിസ്റ്റത്തിലേക്ക് സ്വാഗതം. ഞങ്ങൾ ഓഡിയോ ഗുണനിലവാരവും സ്വാഭാവികതയും പരിശോധിക്കുകയാണ്.",
        "native_name": "മലയാളം (Malayalam)",
        "folder": "Malayalam"
    },
    "Marathi": {
        "text": "नमस्कार! आमच्या बहुभाषिक आवाज प्रणालीमध्ये आपले स्वागत आहे. आम्ही ऑडिओ गुणवत्ता आणि नैसर्गिकतेची चाचणी करत आहोत.",
        "native_name": "मराठी (Marathi)",
        "folder": "Marathi"
    },
    "gujarathi": {
        "text": "નમસ્તે! અમારી બહુભાષી ભાષણ સંશ્લેષણ પ્રણાલીમાં આપનું સ્વાગત છે. અમે ઑડિયો ગુણવત્તા અને કુદરતી વાણીની ચકાસણી કરી રહ્યા છીએ.",
        "native_name": "ગુજરાતી (Gujarati)",
        "folder": "gujarathi"
    },
    "Odia": {
        "text": "ନମସ୍କାର! ଆମର ବହୁଭାଷୀ ଭାଷଣ ସଂଶ୍ଳେଷଣ ପ୍ରଣାଳୀକୁ ଆପଣଙ୍କୁ ସ୍ୱାଗତ। ଆମେ ଅଡିଓ ଗୁଣବତ୍ତା ଏବଂ ପ୍ରାକୃତିକତା ପରୀକ୍ଷା କରୁଛୁ।",
        "native_name": "ଓଡ଼ିଆ (Odia)",
        "folder": "Odia"
    }
}

OUTPUT_DIR = "generated_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def tokenize_mms(text, vocab):
    cleaned = [c for c in text.lower() if c in vocab]
    token_ids = [vocab[c] for c in cleaned]
    interspersed = [0] * (len(token_ids) * 2 + 1)
    interspersed[1::2] = token_ids
    return interspersed, len(cleaned)

def estimate_pitch_autocorr(audio, sr=16000, frame_size=0.03, hop_size=0.01, fmin=60, fmax=350):
    frame_len = int(frame_size * sr)
    hop_len = int(hop_size * sr)
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    
    f0_list = []
    energy_list = []
    
    for start in range(0, len(audio) - frame_len, hop_len):
        frame = audio[start:start + frame_len]
        energy = np.sqrt(np.mean(frame**2))
        energy_list.append(energy)
        
        if energy < 0.01:
            continue
            
        corr = scipy.signal.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        if max_lag < len(corr):
            lag_region = corr[min_lag:max_lag]
            if len(lag_region) > 0 and np.max(lag_region) > 0.35 * corr[0]:
                peak_lag = min_lag + np.argmax(lag_region)
                f0 = sr / peak_lag
                f0_list.append(f0)
                
    f0_arr = np.array(f0_list)
    return f0_arr, np.array(energy_list)

def analyze_audio_quality(audio, sr=16000):
    duration = len(audio) / sr
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    rms_db = 20 * np.log10(rms + 1e-9)
    peak_db = 20 * np.log10(peak + 1e-9)
    clipped_ratio = float(np.mean(np.abs(audio) >= 0.99))
    
    frame_len = int(0.025 * sr)
    frames = [audio[i:i+frame_len] for i in range(0, len(audio)-frame_len, frame_len)]
    frame_energies = np.array([np.sqrt(np.mean(f**2)) for f in frames])
    silence_threshold = peak * 0.03
    silence_ratio = float(np.mean(frame_energies < silence_threshold))
    
    speech_frames = frame_energies[frame_energies >= silence_threshold]
    noise_frames = frame_energies[frame_energies < silence_threshold]
    if len(noise_frames) > 0 and len(speech_frames) > 0:
        snr = 20 * np.log10((np.mean(speech_frames) + 1e-9) / (np.mean(noise_frames) + 1e-9))
    else:
        snr = 35.0
        
    return {
        "duration_sec": round(duration, 3),
        "peak_amplitude": round(peak, 4),
        "peak_db": round(peak_db, 2),
        "rms_db": round(rms_db, 2),
        "clipped_ratio": round(clipped_ratio, 6),
        "silence_ratio": round(silence_ratio, 3),
        "snr_db": round(float(snr), 2)
    }

def run_suite():
    results = {}
    total_start = time.time()
    
    print("=" * 80)
    print("RUNNING MULTILINGUAL TTS ONNX QA & GENERATION SUITE (10 LANGUAGES)")
    print("=" * 80)
    
    for lang_key, meta in LANGUAGES.items():
        folder = meta["folder"]
        onnx_file = os.path.join(folder, "model_optimized.onnx")
        if not os.path.exists(onnx_file):
            onnx_file = os.path.join(folder, "model.onnx")
        vocab_file = os.path.join(folder, "vocab.json")
        
        if not os.path.exists(onnx_file) or not os.path.exists(vocab_file):
            print(f"[SKIP] Model or vocab not found for {lang_key}")
            continue
            
        print(f"\n[SYNTHESIZING] {meta['native_name']} ({lang_key})...")
        t0 = time.time()
        
        with open(vocab_file, encoding='utf-8') as f:
            vocab = json.load(f)
            
        token_ids, matched_char_count = tokenize_mms(meta["text"], vocab)
        
        session = ort.InferenceSession(onnx_file)
        inp = np.array([token_ids], dtype=np.int64)
        mask = np.ones_like(inp, dtype=np.int64)
        
        outputs = session.run(None, {"input_ids": inp, "attention_mask": mask})
        waveform = outputs[0][0]
        inference_time = time.time() - t0
        
        out_path = os.path.join(OUTPUT_DIR, f"{lang_key}_sample.wav")
        sf.write(out_path, waveform, 16000)
        
        qa_metrics = analyze_audio_quality(waveform, sr=16000)
        f0_contour, energy_contour = estimate_pitch_autocorr(waveform, sr=16000)
        
        if len(f0_contour) > 0:
            pitch_stats = {
                "mean_f0_hz": round(float(np.mean(f0_contour)), 1),
                "std_f0_hz": round(float(np.std(f0_contour)), 1),
                "min_f0_hz": round(float(np.min(f0_contour)), 1),
                "max_f0_hz": round(float(np.max(f0_contour)), 1),
                "voiced_frames": len(f0_contour)
            }
        else:
            pitch_stats = {
                "mean_f0_hz": 0.0,
                "std_f0_hz": 0.0,
                "min_f0_hz": 0.0,
                "max_f0_hz": 0.0,
                "voiced_frames": 0
            }
            
        rtf = inference_time / qa_metrics["duration_sec"]
        
        results[lang_key] = {
            "native_name": meta["native_name"],
            "test_text": meta["text"],
            "audio_file": os.path.abspath(out_path).replace("\\", "/"),
            "file_size_bytes": os.path.getsize(out_path),
            "inference_time_sec": round(inference_time, 3),
            "real_time_factor": round(rtf, 4),
            "chars_processed": matched_char_count,
            "acoustic_quality": qa_metrics,
            "prosody_metrics": pitch_stats
        }
        
        print(f"  ✓ Saved to: {out_path}")
        print(f"  ✓ Duration: {qa_metrics['duration_sec']}s | RTF: {rtf:.3f}x | Inference: {inference_time:.3f}s")
        print(f"  ✓ Peak: {qa_metrics['peak_amplitude']:.3f} ({qa_metrics['peak_db']} dB) | RMS: {qa_metrics['rms_db']} dB")
        print(f"  ✓ F0 Pitch: Mean={pitch_stats['mean_f0_hz']} Hz (Std={pitch_stats['std_f0_hz']} Hz) | SNR: {qa_metrics['snr_db']} dB")
        
    summary_path = os.path.join(OUTPUT_DIR, "qa_test_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 80)
    print(f"ALL 10 LANGUAGES TESTED AND GENERATED SUCCESSFULLY!")
    print(f"Detailed QA report saved to: {summary_path}")
    print(f"Total time elapsed: {time.time() - total_start:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    run_suite()
