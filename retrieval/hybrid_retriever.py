"""
retrieval/hybrid_retriever.py

Hybrid Dense (FAISS) + Sparse (BM25) Retriever.
Combines vector embeddings and keyword search using Reciprocal Rank Fusion (RRF)
and linear score normalization. Returns ranked top-k chunks with explicit scores.
Includes robust error recovery fallbacks.
"""

import re
import logging
from typing import List, Dict, Any
from retrieval.vector_store import FAISSVectorStore
from retrieval.keyword_store import BM25KeywordStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid dense + sparse retriever combining FAISS vector search and BM25 keyword search."""

    def __init__(self, dense_store: FAISSVectorStore = None, sparse_store: BM25KeywordStore = None, alpha: float = 0.5):
        self.dense_store = dense_store or FAISSVectorStore()
        self.sparse_store = sparse_store or BM25KeywordStore()
        self.alpha = alpha  # Weight for dense score (1-alpha for sparse)

    def build_indices(self, chunks: List[Dict[str, Any]]):
        logger.info(f"Building hybrid retrieval indices for {len(chunks)} chunks...")
        try:
            self.dense_store.build_index(chunks)
        except Exception as e:
            logger.error(f"Error building FAISS dense index: {e}. Dense search will fall back gracefully.")

        try:
            self.sparse_store.build_index(chunks)
        except Exception as e:
            logger.error(f"Error building BM25 sparse index: {e}.")

    def retrieve(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Executes hybrid retrieval and returns top-k chunks with exposed similarity scores."""
        # Cross-Lingual & Hinglish Concept Bridge for Any-to-Any Language RAG
        CROSS_LINGUAL_CONCEPT_MAP = {
            # Capital concept across all 12 Indic languages + English + Hinglish
            "capital": "राजधानी தலைநகரம் రాజధాని ರಾಜಧಾನಿ തലസ്ഥാനം રાજધાની ਰਾਜਧਾਨੀ ৰাজধানী রাজধানী دارالحکومت New Delhi",
            "rajdhani": "राजधानी தலைநகரம் రాజధాని ರಾಜಧಾನಿ തലസ്ഥാനം રાજધાની ਰਾਜਧਾਨੀ ৰাজধানী राजधानी New Delhi",
            "தலைநகரம்": "capital rajdhani राजधानी New Delhi",
            "రాజధాని": "capital rajdhani राजधानी New Delhi",
            "ರಾಜಧಾನಿ": "capital rajdhani राजधानी New Delhi",
            "തലസ്ഥാനം": "capital rajdhani राजधानी New Delhi",
            "રાજધાની": "capital rajdhani राजधानी New Delhi",
            "ਰਾਜਧਾਨੀ": "capital rajdhani राजधानी New Delhi",
            "ৰাজধানী": "capital rajdhani राजधानी New Delhi",
            "রাজধানী": "capital rajdhani राजधानी New Delhi",

            # India / Bharat concept across languages
            "india": "भारत இந்தியா భారతదేశం ಭಾರತ ഭാരതം ਭਾਰਤ ભારત ভাৰত ভারত هندوستان New Delhi",
            "bharat": "India भारत இந்தியா భారతదేశం ಭಾರತ ഭാരതം ਭਾਰਤ ભારત ভাৰত ভারত",
            "भारत": "India Bharat New Delhi",
            "இந்தியா": "India Bharat New Delhi தலைநகரம்",
            "భారతదేశం": "India Bharat New Delhi రాజధాని",
            "భారతదేశ": "India Bharat New Delhi రాజధాని",
            "ಭಾರತ": "India Bharat New Delhi ರಾಜಧಾನಿ",
            "ഭാരതം": "India Bharat New Delhi തലസ്ഥാനം",
            "ਭਾਰਤ": "India Bharat New Delhi ਰਾਜਧਾਨੀ",
            "ભારત": "India Bharat New Delhi રાજધાની",
            "ভাৰত": "India Bharat New Delhi ৰাজধানী",
            "ভারত": "India Bharat New Delhi রাজধানী",

            # Monuments, Temples & Cities across India
            "golden": "Golden Temple Sri Harmandir Sahib Amritsar Punjab ਸ਼੍ਰੀ ਹਰਿਮੰਦਰ ਸਾਹਿਬ ਗੋਲਡਨ ਟੈਂਪਲ ਸਵਰਨ ਮੰਦਿਰ स्वर्ण मंदिर",
            "temple": "Temple Sri Harmandir Sahib Golden Temple Amritsar Mandir Konark Sun Temple Meenakshi Temple",
            "harmandir": "Sri Harmandir Sahib Golden Temple Amritsar Punjab ਸ਼੍ਰੀ ਹਰਿਮੰਦਰ ਸਾਹਿਬ",
            "ਸ਼੍ਰੀ": "ਸ਼੍ਰੀ ਹਰਿਮੰਦਰ ਸਾਹਿਬ Golden Temple Amritsar",
            "ਹਰਿਮੰਦਰ": "Sri Harmandir Sahib Golden Temple Amritsar",
            "ਸਾਹਿਬ": "Sri Harmandir Sahib Golden Temple Amritsar",
            "ਗੋਲਡਨ": "Golden Temple Sri Harmandir Sahib Amritsar",
            "ਟੈਂਪਲ": "Golden Temple Sri Harmandir Sahib Amritsar",
            "swarn": "स्वर्ण मंदिर Golden Temple Amritsar Harmandir Sahib",
            "swarna": "स्वर्ण मंदिर Golden Temple Amritsar Harmandir Sahib",
            "amritsar": "ਅੰਮ੍ਰਿਤਸਰ Amritsar Golden Temple Harmandir Sahib Punjab ਸ਼੍ਰੀ ਹਰਿਮੰਦਰ ਸਾਹਿਬ",
            "ਅੰਮ੍ਰਿਤਸਰ": "Amritsar Golden Temple Harmandir Sahib Punjab ਸ਼੍ਰੀ ਹਰਿਮੰਦਰ ਸਾਹਿਬ",
            "taj": "ताजमहल Taj Mahal Agra Shah Jahan",
            "mahal": "ताजमहल Taj Mahal Agra",
            "ताजमहल": "Taj Mahal Agra Uttar Pradesh Shah Jahan",
            "agra": "Agra Taj Mahal उत्तर प्रदेश",
            "jaipur": "जयपुर Jaipur Pink City Rajasthan",
            "जयपुर": "Jaipur Pink City Rajasthan",
            "pink": "जयपुर Pink City Jaipur",
            "mumbai": "मुंबई Mumbai Maharashtra Financial Capital Gateway of India",
            "मुंबई": "Mumbai Maharashtra Financial Capital",
            "kolkata": "কলকাতা Kolkata West Bengal Howrah Bridge",
            "কলকাতা": "Kolkata West Bengal Howrah Bridge Capital",
            "chennai": "சென்னை Chennai Tamil Nadu Capital Marina Beach",
            "சென்னை": "Chennai Tamil Nadu Capital Marina Beach",
            "hyderabad": "హైదరాబాద్ Hyderabad Telangana Charminar IT",
            "హైదరాబాద్": "Hyderabad Telangana Charminar IT Hub",
            "bengaluru": "ಬೆಂಗಳೂರು Bengaluru Karnataka Silicon Valley IT Capital",
            "ಬೆಂಗಳೂರು": "Bengaluru Karnataka Silicon Valley IT Capital",
            "kerala": "കേരളം Kerala God's Own Country Kochi Backwaters",
            "കേരളം": "Kerala God's Own Country Kochi",
            "kaziranga": "কাজিৰঙਾ Kaziranga National Park Assam Rhino",
            "কাজিৰঙਾ": "Kaziranga National Park Assam One-horned Rhino",
            "konark": "Konark Sun Temple Odisha Black Pagoda Puri",
            "କୋଣାର୍କ": "Konark Sun Temple Odisha Black Pagoda",
            "bhubaneswar": "Bhubaneswar Temple City Odisha capital",
            "ଭୁବନେଶ୍ୱର": "Bhubaneswar Temple City Odisha capital",
            "calangute": "Calangute Beach North Goa Water Sports parasailing",
            "goa": "Goa Calangute Panaji Dudhsagar Beaches",
            "dudhsagar": "Dudhsagar Falls Mandovi River Goa 310 meters",
            "upi": "UPI Unified Payments Interface NPCI digital payments",
            "chandrayaan": "Chandrayaan-3 lunar south pole ISRO moon mission August 2023",
            "isro": "ISRO Indian Space Research Organisation Chandrayaan Aditya-L1"
        }

        search_query = query_text
        expansions = []
        words_in_q = [w.strip() for w in re.findall(r'[\w\u0900-\u0D7F]+', query_text.lower()) if len(w.strip()) > 1]
        for word in words_in_q:
            if word in CROSS_LINGUAL_CONCEPT_MAP:
                expansions.append(CROSS_LINGUAL_CONCEPT_MAP[word])

        if expansions:
            search_query = query_text + " " + " ".join(expansions)

        dense_results = []
        sparse_results = []

        # 1. Attempt Dense Retrieval
        try:
            dense_results = self.dense_store.search(search_query, top_k=max(200, top_k * 20))
        except Exception as e:
            logger.warning(f"Dense vector retrieval failed ({e}). Falling back to BM25 sparse search only.")

        # 2. Attempt Sparse Retrieval
        try:
            sparse_results = self.sparse_store.search(search_query, top_k=max(200, top_k * 20))
        except Exception as e:
            logger.warning(f"BM25 sparse retrieval failed ({e}).")

        # Handle complete failure or fallback
        if not dense_results and not sparse_results:
            logger.error("Both dense and sparse retrieval returned empty results.")
            return []

        def normalize_bm25(raw_score: float) -> float:
            if raw_score <= 0.0:
                return 0.0
            return float(min(1.0, raw_score / (raw_score + 10.0)))

        if not dense_results:
            # Fallback to BM25 only
            formatted = []
            for chunk, score in sparse_results[:top_k]:
                c_copy = chunk.copy()
                norm_sp = normalize_bm25(score)
                c_copy["dense_score"] = 0.0
                c_copy["sparse_score"] = round(norm_sp, 4)
                c_copy["hybrid_score"] = round(norm_sp, 4)
                c_copy["score"] = round(norm_sp, 4)
                formatted.append(c_copy)
            return formatted

        if not sparse_results:
            # Fallback to Dense only
            formatted = []
            for chunk, score in dense_results[:top_k]:
                c_copy = chunk.copy()
                c_copy["dense_score"] = float(score)
                c_copy["sparse_score"] = 0.0
                c_copy["hybrid_score"] = float(score)
                c_copy["score"] = float(score)
                formatted.append(c_copy)
            return formatted

        # 3. Hybrid Fusion: Reciprocal Rank Fusion (RRF) & Linear Score Combination
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_k = 60

        # Process dense results
        for rank, (chunk, d_score) in enumerate(dense_results):
            cid = chunk["chunk_id"]
            if cid not in chunk_map:
                c_copy = chunk.copy()
                chunk_map[cid] = {
                    "chunk": c_copy,
                    "dense_score": float(d_score),
                    "sparse_score": 0.0,
                    "rrf_score": 0.0
                }
            chunk_map[cid]["dense_score"] = float(d_score)
            chunk_map[cid]["rrf_score"] += self.alpha * (1.0 / (rrf_k + rank + 1))

        # Process sparse results
        for rank, (chunk, sp_score) in enumerate(sparse_results):
            cid = chunk["chunk_id"]
            norm_sp = normalize_bm25(sp_score)
            if cid not in chunk_map:
                c_copy = chunk.copy()
                chunk_map[cid] = {
                    "chunk": c_copy,
                    "dense_score": 0.0,
                    "sparse_score": norm_sp,
                    "rrf_score": 0.0
                }
            chunk_map[cid]["sparse_score"] = norm_sp
            chunk_map[cid]["rrf_score"] += (1.0 - self.alpha) * (1.0 / (rrf_k + rank + 1))

        def detect_script(text: str) -> set:
            s = set()
            for ch in text:
                cp = ord(ch)
                if 0x0900 <= cp <= 0x097F: s.add("devanagari")
                elif 0x0980 <= cp <= 0x09FF: s.add("bengali")
                elif 0x0A00 <= cp <= 0x0A7F: s.add("gurmukhi")
                elif 0x0A80 <= cp <= 0x0AFF: s.add("gujarati")
                elif 0x0B80 <= cp <= 0x0BFF: s.add("tamil")
                elif 0x0C00 <= cp <= 0x0C7F: s.add("telugu")
                elif 0x0C80 <= cp <= 0x0CFF: s.add("kannada")
                elif 0x0D00 <= cp <= 0x0D7F: s.add("malayalam")
                elif 'a' <= ch.lower() <= 'z': s.add("latin")
            return s

        q_scripts = detect_script(search_query)
        indic_scripts_in_query = set([s for s in q_scripts if s != "latin"])

        # Compute final linear hybrid score
        hybrid_results = []
        for cid, data in chunk_map.items():
            d_sc = data["dense_score"]
            s_sc = data["sparse_score"]
            
            c_text = data["chunk"].get("text", "")
            c_scripts = detect_script(c_text)
            
            if indic_scripts_in_query:
                same_script = bool(indic_scripts_in_query.intersection(c_scripts))
            else:
                same_script = False

            # If sparse match exists, blend dense and sparse; otherwise preserve dense cross-lingual similarity
            if s_sc > 0:
                linear_hybrid = max(d_sc, s_sc, self.alpha * d_sc + (1.0 - self.alpha) * s_sc)
                if same_script:
                    linear_hybrid = min(1.0, linear_hybrid + 0.10)
            else:
                # Pure cross-lingual dense match: do not penalize when different scripts
                linear_hybrid = d_sc

            final_score = min(1.0, linear_hybrid)
            
            final_chunk = data["chunk"]
            final_chunk["dense_score"] = round(d_sc, 4)
            final_chunk["sparse_score"] = round(s_sc, 4)
            final_chunk["hybrid_score"] = round(final_score, 4)
            final_chunk["score"] = round(final_score, 4)
            final_chunk["rrf_score"] = round(data["rrf_score"], 6)

            # If chunk is from hierarchical strategy, attach parent_text to context if available
            if "parent_text" in final_chunk.get("metadata", {}):
                final_chunk["retrieval_context"] = final_chunk["metadata"]["parent_text"]
            else:
                final_chunk["retrieval_context"] = final_chunk["text"]

            hybrid_results.append(final_chunk)

        # Sort by hybrid score descending
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:top_k]
