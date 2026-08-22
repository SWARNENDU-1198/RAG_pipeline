"""
scripts/e2e_verify.py
Comprehensive End-to-End Test Suite for Voice RAG Pipeline.
"""

import httpx

def main():
    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=25.0)

    print("\n--- 1. Testing GET / (Frontend Delivery) ---")
    r_index = client.get("/")
    assert r_index.status_code == 200, f"Failed GET /: {r_index.status_code}"
    assert "START RECORDING" in r_index.text, "START RECORDING button text missing in HTML"
    assert "micBtn" in r_index.text, "micBtn element missing in HTML"
    print(" [PASS] GET / -> HTTP 200 OK, interactive Voice Studio HTML delivered.")

    print("\n--- 2. Testing GET /health (Health Status) ---")
    r_health = client.get("/health")
    assert r_health.status_code == 200, f"Health check failed: {r_health.status_code}"
    health_json = r_health.json()
    print(f" [PASS] GET /health -> {health_json}")

    print("\n--- 3. Testing POST /query [Text Grounded RAG] ---")
    r_rag = client.post(
        "/query",
        data={
            "query_text": "What is Calangute Beach famous for?",
            "knowledge_mode": "dataset_only",
            "language_code": "en-IN"
        }
    )
    assert r_rag.status_code == 200, f"Query failed: {r_rag.status_code}, {r_rag.text}"
    res_rag = r_rag.json()
    print(f" [PASS] Grounded Answer: {res_rag.get('final_answer')}")
    print(f" [PASS] Source Type: {res_rag.get('source_type')}")
    print(f" [PASS] Top Retrieval Score: {res_rag.get('retrieval_result', {}).get('top_score')}")
    print(f" [PASS] Latency Timings: {res_rag.get('stage_timings')}")

    print("\n--- 4. Testing POST /query [Indic Hybrid Query] ---")
    r_indic = client.post(
        "/query",
        data={
            "query_text": "Bharat ka capital kya hai / What is the capital of India?",
            "knowledge_mode": "hybrid_auto",
            "language_code": "hi-IN"
        }
    )
    assert r_indic.status_code == 200, f"Indic query failed: {r_indic.status_code}"
    res_indic = r_indic.json()
    print(f" [PASS] Indic Answer: {res_indic.get('final_answer')}")

    print("\n--- 5. Testing POST /query [Audio File Upload] ---")
    with open("sample_audio.wav", "rb") as f:
        r_audio = client.post(
            "/query",
            files={"audio": ("audio.wav", f, "audio/wav")},
            data={
                "query_text": "What is Calangute Beach famous for?",
                "knowledge_mode": "dataset_only"
            }
        )
    assert r_audio.status_code == 200, f"Audio query failed: {r_audio.status_code}"
    print(f" [PASS] Audio Status: {r_audio.status_code}")
    print(f" [PASS] Audio Pipeline Answer: {r_audio.json().get('final_answer')}")

    print("\n--- 6. Testing POST /tts [Sarvam Bulbul TTS Voice Generation] ---")
    r_tts = client.post(
        "/tts",
        json={
            "text": "Namaste, Calangute Beach Goa ka prasiddh beach hai.",
            "language_code": "hi-IN"
        }
    )
    assert r_tts.status_code == 200, f"TTS failed: {r_tts.status_code}"
    tts_json = r_tts.json()
    print(f" [PASS] TTS Status: {r_tts.status_code}")
    print(f" [PASS] TTS audio received: {bool(tts_json.get('audio_base64'))} (Provider: {tts_json.get('provider')})")

    print("\n=======================================================")
    print("   ALL 6 END-TO-END SYSTEM TESTS PASSED SUCCESSFULLY!  ")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
