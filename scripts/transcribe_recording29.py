from faster_whisper import WhisperModel

MODEL = "medium.en"
PROMPT = "Platinum Fashion Mall Bangkok. Bringing Fresh Fashion to You. Shopping, fashion, clothing, shoes, accessories, gifts, wholesale stores."

model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=4)

for audio in ["original.wav", "enhanced.wav"]:
    print(f"\n===== TRANSCRIPT: {audio} =====")
    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        best_of=5,
        vad_filter=False,
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=PROMPT,
        temperature=0.0,
        no_speech_threshold=0.95,
        log_prob_threshold=-1.5,
    )
    for seg in segments:
        text = seg.text.strip()
        print(f"[{seg.start:06.2f}-{seg.end:06.2f}] {text}")
        if seg.words:
            words = " ".join(f"{w.word.strip()}<{w.start:.2f}-{w.end:.2f}>" for w in seg.words)
            print("WORDS:", words)
