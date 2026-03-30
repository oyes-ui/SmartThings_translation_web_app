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
    GROUP_A_SHEETS,
    GROUP_A_KEY,
    GROUP_B_KEY,
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


    def _get_collection(self, target_lang: str) -> chromadb.Collection:
        """타겟 언어에 따라 KR 또는 US 컬렉션 선택"""
        if target_lang in GROUP_A_SHEETS:
            return self._col_kr
        return self._col_us

    def retrieve(
        self,
        source_text: str,
        target_lang: str,
        n_results: int = 2,
        exclude_same_source: bool = True
    ) -> list[dict]:
        """
        유사 번역 사례 검색.

        Args:
            source_text: 현재 번역할 원문 텍스트
            target_lang: 타겟 언어 시트명 (예: "DE(독일)", "JA(일본)")
            n_results: 반환할 사례 수 (기본 2)
            exclude_same_source: 원문이 동일한 결과 제외

        Returns:
            [{"source": ..., "target": ..., "section_code": ..., "story_id": ...}, ...]
        """
        if not self._available:
            return []

        try:
            col = self._get_collection(target_lang)
            query_embedding = self._embed_query(source_text)

            results = col.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results + 2, col.count()),  # 여유분 확보
                where={"target_lang": target_lang} if col.count() > 0 else None,
                include=["documents", "metadatas", "distances"]
            )

            if not results["ids"][0]:
                return []

            examples = []
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i]
                source = results["documents"][0][i]
                distance = results["distances"][0][i]

                # 동일 원문 제외 (self-match)
                if exclude_same_source and source.strip() == source_text.strip():
                    continue

                # 거리가 너무 먼 경우 제외 (유사도 임계값: 코사인 거리 < 0.8)
                if distance > 0.8:
                    continue

                examples.append({
                    "source": source,
                    "target": meta.get("target_text", ""),
                    "section_code": meta.get("section_code", ""),
                    "story_id": meta.get("story_id", ""),
                    "similarity_score": round(1 - distance, 3)
                })

                if len(examples) >= n_results:
                    break

            return examples

        except Exception as e:
            print(f"[RAG] 검색 오류: {e}")
            return []

    def format_for_prompt(
        self,
        source_text: str,
        target_lang: str,
        n_results: int = 2
    ) -> str:
        """
        프롬프트에 주입할 형식으로 RAG 예시 포맷팅.

        반환 예:
        [Translation Memory Examples]
        Example 1 (similarity: 0.92, section: //section_001_1):
          Source: "AI가 관리해 주는 전기 요금"
          Translation: "Von KI verwaltete Stromrechnungen"

        Example 2 (similarity: 0.87, section: //section_001_2):
          Source: "에너지를 더 똑똑하게"
          Translation: "Energie intelligenter nutzen"
        """
        examples = self.retrieve(source_text, target_lang, n_results)
        if not examples:
            return ""

        lines = ["[Translation Memory Examples]"]
        for i, ex in enumerate(examples, 1):
            lines.append(
                f"Example {i} (similarity: {ex['similarity_score']}, "
                f"section: {ex['section_code']}):"
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
