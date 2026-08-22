"""
benchmarking/test_suite_demo.py

Comprehensive Interactive & Batch Verification Test Suite.
Demonstrates end-to-end question answering and latency across multiple languages and domains.
"""

import os
import sys
import json
import time
import io

# Force UTF-8 output encoding for terminal printing
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.orchestrator import VoiceRAGPipelineOrchestrator
from pipeline.schemas import AudioInput


def run_test_suite():
    print("=" * 95)
    print("      MULTILINGUAL VOICE RAG PIPELINE - E2E DATASET TEST SUITE (< 200ms TARGET)")
    print("=" * 95)

    orchestrator = VoiceRAGPipelineOrchestrator()
    orchestrator.initialize_index()

    test_queries = [
        # Hindi & Hinglish
        ("Bharat ka capital kya hai?", "Hindi / Hinglish (Geography)"),
        ("भारत की राजधानी क्या है?", "Hindi (Geography)"),
        ("Taj Mahal kisne banwaya tha?", "Hindi / Hinglish (Heritage)"),
        ("जयपुर को किस नाम से जाना जाता है?", "Hindi (Pink City)"),
        ("Bharat ka rashtriya gaan kya hai?", "Hindi (Anthem)"),
        ("गंगा नदी का उद्गम कहाँ से होता है?", "Hindi (River Source)"),

        # Bengali
        ("পশ্চিমবঙ্গের রাজধানী কোনটি?", "Bengali (Capital)"),
        ("রবীন্দ্রনাথ ঠাকুর কেন নোবেল पुरस्कार পেয়েছিলেন?", "Bengali (Nobel Prize)"),
        ("সুন্দরবন বিশ্বের বৃহত্তম ম্যানগ্রোভ বন এবং এটি কিসের জন্য বিখ্যাত?", "Bengali (Sundarbans)"),

        # Tamil
        ("தமிழ்நாட்டின் தலைநகரம் எது?", "Tamil (Capital)"),
        ("மதுரையில் உள்ள புகழ்பெற்ற கோவில் எது?", "Tamil (Meenakshi Temple)"),
        ("தஞ்சாவூர் பெரிய கோவிலை கட்டியவர் யார்?", "Tamil (Brihadisvara Temple)"),

        # Telugu
        ("హైదరాబాద్ నగరం దేనికి ప్రசிద్ధి చెందింది?", "Telugu (IT & Charminar)"),
        ("తిరుపతిలో ఉన్న ప్రசிద్ధ దేవాలయం ఏది?", "Telugu (Tirupati Temple)"),
        ("ఆంధ్రప్రదేశ్ శాస్త్రీయ నృత్యం ఏది?", "Telugu (Kuchipudi)"),

        # Kannada
        ("ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?", "Kannada (Bengaluru)"),
        ("ಹಂಪಿ ಯಾವುದಕ್ಕೆ ಪ್ರಸಿದ್ಧವಾಗಿದೆ?", "Kannada (Hampi UNESCO)"),

        # Malayalam
        ("കേരളം എങ്ങനെ അറിയപ്പെടുന്നു?", "Malayalam (God's Own Country)"),
        ("കഥകളി കേരളത്തിന്റെ പരമ്പരാഗത നൃത്യരൂപം ഏതാണ്?", "Malayalam (Kathakali)"),

        # Marathi
        ("महाराष्ट्राची आर्थिक राजधानी कोणती?", "Marathi (Mumbai)"),
        ("मराठा साम्राज्याची स्थापना कोणी केली?", "Marathi (Shivaji Maharaj)"),

        # Gujarati
        ("ગુજરાતની રાજધાની કઈ છે?", "Gujarati (Gandhinagar)"),
        ("સ્ટેચ્યુ ઓફ યુનિટી કોની પ્રતિમા છે?", "Gujarati (Statue of Unity)"),

        # Punjabi
        ("ਸ਼੍ਰੀ ਹਰਿਮੰਦਰ ਸਾਹਿਬ ਕਿੱਥੇ ਸਥਿਤ ਹੈ?", "Punjabi (Golden Temple)"),
        ("ਪੰਜਾਬ ਦਾ ਪ੍ਰਸਿੱਧ ਲੋਕ ਨਾਚ ਕਿਹੜਾ ਹੈ?", "Punjabi (Bhangra)"),

        # Assamese & Odia
        ("কাজিৰঙা ৰাষ্ট্ৰীয় উদ্যান কিয় বিখ্যাত?", "Assamese (Kaziranga)"),
        ("ଓଡ଼ିଶାର ରାଜଧାନୀ କଣ?", "Odia (Bhubaneswar)"),

        # Goa Tourism & Heritage
        ("What is Calangute Beach famous for?", "English (Goa Beach)"),
        ("Where are the remains of Saint Francis Xavier stored?", "English (Goa Basilica)"),
        ("Which river forms the Dudhsagar Falls in Goa?", "English (Goa Waterfall)"),

        # Science & Technology
        ("When did Chandrayaan-3 land on the moon?", "English (Space / ISRO)"),
        ("What is the objective of Aditya-L1 launched by ISRO?", "English (Solar Mission)"),
        ("Who developed the UPI payment system in India?", "English (Fintech / NPCI)"),
        ("Who was Aryabhata?", "English (Mathematics)"),
        ("What is the function of the Reserve Bank of India (RBI)?", "English (Economy / Central Bank)"),

        # Architecture & Voice AI
        ("What languages and capabilities does Sarvam AI support?", "English (Voice AI)"),
        ("What is FAISS used for in vector retrieval?", "English (Dense Indexing)"),
        ("What is hybrid search in RAG systems?", "English (Hybrid Search)")
    ]

    print(f"\nRunning {len(test_queries)} Multilingual & Domain Test Queries...\n")
    print(f"{'#':<3} | {'Domain / Language':<30} | {'E2E Latency':<12} | {'Retrieval':<10} | {'Status':<8}")
    print("-" * 95)

    latencies = []

    for i, (query, category) in enumerate(test_queries, 1):
        inp = AudioInput(text_override=query)
        res = orchestrator.run(inp)
        latencies.append(res.total_latency_ms)

        status_icon = "✅ PASS" if res.total_latency_ms < 200 and not res.is_refused else "❌ FAIL"
        ret_ms = f"{res.stage_timings.get('retrieval', 0.0):.1f}ms"
        tot_ms = f"{res.total_latency_ms:.2f}ms"

        print(f"{i:<3} | {category:<30} | {tot_ms:<12} | {ret_ms:<10} | {status_icon:<8}")
        print(f"    Q: {query}")
        print(f"    A: {res.final_answer}")
        print()

    avg_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    min_lat = min(latencies)

    print("=" * 95)
    print(f"SUMMARY: {len(test_queries)} Queries Tested | Avg Latency: {avg_lat:.2f}ms | Min: {min_lat:.2f}ms | Max: {max_lat:.2f}ms")
    print(f"All Queries Executed Under 200ms: {'✅ YES' if max_lat < 200 else '❌ NO'}")
    print("=" * 95)


if __name__ == "__main__":
    run_test_suite()
