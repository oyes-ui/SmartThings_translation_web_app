# -*- coding: utf-8 -*-
"""
rag_retriever.py
SmartThings 번역 RAG 검색 모듈

번역 요청 시 유사 과거 번역 사례를 ChromaDB에서 검색하여 프롬프트용 문자열로 반환합니다.

사용법 (CLI 테스트):
  python rag_retriever.py --test "AI goes to sleep" --lang "DE(독일)"
  python rag_retriever.py --test "AI 절약 모드 설정" --lang "JA(일본)"
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── rag_db_builder 상수 재사용 ──────────────────────────────────────────────
from rag_db_builder import (
    normalize_text,
    get_collection_name,
    GROUP_A_SHEETS,
    COLLECTION_KR,
    COLLECTION_US,
    EMBEDDING_MODEL,
    CHROMA_DIR,
    get_gemini_client,
    get_chroma_collections,
)
import chromadb
from google.genai import types


# ─── 검색 클라이언트 (싱글턴 패턴) ───────────────────────────────────────────
_retriever_instance = None


class RagRetriever:
    """
    RAG 검색기. DB 없으면 graceful fallback (빈 결과 반환).
    """

    def __init__(self):
        self._gemini = None
        self._col_kr = None
        self._col_us = None
        self._available = False

        try:
            self._gemini = get_gemini_client()
            _, self._col_kr, self._col_us = get_chroma_collections()
            kr_count = self._col_kr.count()
            us_count = self._col_us.count()
            if kr_count + us_count > 0:
                self._available = True
                print(f"[RAG] DB 로드 완료 — KR:{kr_count}건, US:{us_count}건")
            else:
                print("[RAG] DB가 비어 있습니다. RAG 기능 비활성화.")
        except Exception as e:
            print(f"[RAG] 초기화 실패 (RAG 없이 동작): {e}")

    def is_available(self) -> bool:
        return self._available

    def _embed_query(self, text: str) -> list[float]:
        """쿼리 텍스트를 벡터로 변환"""
        response = self._gemini.models.embed_content(
            model=f"models/{EMBEDDING_MODEL}",
            contents=[text],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values


    def _get_collection(self, source_lang: str) -> chromadb.Collection:
        """source_lang에 따라 KR 또는 US 컬렉션 선택"""
        col_name = get_collection_name(source_lang)
        if col_name == COLLECTION_KR:
            return self._col_kr
        return self._col_us

    def retrieve(
        self,
        source_text: str,
        target_lang: str,
        source_lang: str = "English",
        n_results: int = 2,
        exclude_same_source: bool = False,
        identity_match_enabled: bool = True
    ) -> list[dict]:
        """
        유사 번역 사례 검색 (2단계: Exact Match -> Semantic Fallback)
        """
        if not self._available:
            return []

        try:
            col = self._get_collection(source_lang)
            norm_query = normalize_text(source_text)
            examples = []

            # Stage 1: Exact Match (Metadata Filter)
            if identity_match_enabled:
                # Build where clause based on target_lang
                where_clause = {"source_text_norm": norm_query}
                if target_lang and target_lang.lower() != "all":
                    where_clause = {"$and": [where_clause, {"target_lang": target_lang}]}

                exact_results = col.get(
                    where=where_clause,
                    limit=n_results,
                    include=["documents", "metadatas"]
                )

                if exact_results["ids"]:
                    for i, doc_id in enumerate(exact_results["ids"]):
                        meta = exact_results["metadatas"][i]
                        source = exact_results["documents"][i]
                        examples.append({
                            "source": source,
                            "target": meta.get("target_text", ""),
                            "section_code": meta.get("section_code", ""),
                            "story_id": meta.get("story_id", ""),
                            "match_type": "exact",
                            "similarity_score": 1.0
                        })

            # Stage 2: Semantic Fallback (만약 결과가 부족하면)
            if len(examples) < n_results:
                query_embedding = self._embed_query(norm_query)
                
                query_where = None
                if col.count() > 0 and target_lang and target_lang.lower() != "all":
                    query_where = {"target_lang": target_lang}
                
                results = col.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results + 2,
                    where=query_where,
                    include=["documents", "metadatas", "distances"]
                )

                if results["ids"][0]:
                    for i, doc_id in enumerate(results["ids"][0]):
                        if len(examples) >= n_results: break
                        
                        # 이미 exact match로 찾은 건 스킵 (ID 중복 방지)
                        if any(ex.get("source") == results["documents"][0][i] for ex in examples):
                            continue
                            
                        meta = results["metadatas"][0][i]
                        source = results["documents"][0][i]
                        distance = results["distances"][0][i]

                        # 거리가 너무 먼 경우 제외 (유사도 임계값: 코사인 거리 < 0.8)
                        if distance > 0.8:
                            continue

                        examples.append({
                            "source": source,
                            "target": meta.get("target_text", ""),
                            "section_code": meta.get("section_code", ""),
                            "story_id": meta.get("story_id", ""),
                            "match_type": "semantic",
                            "similarity_score": round(1 - distance, 3)
                        })

            return examples

        except Exception as e:
            print(f"[RAG] 검색 오류: {e}")
            return []

    def format_for_prompt(
        self,
        source_text: str,
        target_lang: str,
        source_lang: str = "English",
        n_results: int = 2,
        identity_match_enabled: bool = True
    ) -> str:
        """
        프롬프트에 주입할 형식으로 RAG 예시 포맷팅.
        """
        examples = self.retrieve(
            source_text, 
            target_lang, 
            source_lang=source_lang, 
            n_results=n_results,
            identity_match_enabled=identity_match_enabled
        )
        if not examples:
            return ""

        lines = ["[Translation Memory Examples]"]
        for i, ex in enumerate(examples, 1):
            match_info = f"match: {ex['match_type']}"
            if ex['match_type'] == "semantic":
                match_info += f", similarity: {ex['similarity_score']}"
            
            lines.append(
                f"Example {i} ({match_info}, section: {ex['section_code']}):"
            )
            lines.append(f'  Source: "{ex["source"]}"')
            lines.append(f'  Translation: "{ex["target"]}"')
            lines.append("")

        return "\n".join(lines).strip()


def get_retriever() -> RagRetriever:
    """싱글턴 RagRetriever 반환 (서버 재시작 없이 전역 재사용)"""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RagRetriever()
    return _retriever_instance


# ─── CLI 테스트 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 검색 테스트")
    parser.add_argument("--test", required=True, help="검색할 원문 텍스트")
    parser.add_argument("--lang", required=True, help="타겟 언어 시트명 (예: 'DE(독일)')")
    parser.add_argument("--n", type=int, default=2, help="반환 결과 수 (기본 2)")
    args = parser.parse_args()

    retriever = RagRetriever()

    if not retriever.is_available():
        print("❌ RAG DB가 없습니다. 먼저 'python rag_db_builder.py --pilot' 을 실행하세요.")
        sys.exit(1)

    print(f"\n🔍 검색 쿼리: \"{args.test}\" → [{args.lang}]")
    print("-" * 60)
    prompt_text = retriever.format_for_prompt(args.test, args.lang, args.n)
    if prompt_text:
        print(prompt_text)
    else:
        print("(결과 없음 — DB에 해당 언어 데이터가 없거나 유사 사례가 없습니다)")
