# -*- coding: utf-8 -*-
"""
glossary_checks.py — 글로서리 매칭/케이싱/브래킷/브랜드 띄어쓰기 검사 로직.

checker_service.py 의 TranslationChecker 에서 분리됨(2026-08-03). 정책 판단
(어떤 규칙을 적용할지)은 prompt_builder.py 의 PromptBuilder 가 갖고 있고, 여기서는
그 정책을 실제 텍스트에 기계적으로 적용하는 결정론적(non-LLM) 로직만 다룬다.

TranslationChecker 는 이 클래스의 인스턴스를 self._glossary_checker 로 갖고,
기존에 이 파일로 옮겨진 메서드와 같은 이름의 얇은 위임 wrapper를 유지한다 —
외부 스크립트 여러 곳이 TranslationChecker의 private 메서드(`_get_glossary_context_as_dict`
등)와 `.glossary` 속성을 직접 참조하기 때문에, 이름과 시그니처를 그대로 지켜야 한다.
"""

from __future__ import annotations

import re

from translation_web_app.prompt_builder import PromptBuilder
from translation_web_app.constraint_resolver import ConstraintResolver, OccurrenceActivationManifest

# Quote-pair styles that can delimit a navigation path across target locales.
# Real DB targets mix these inconsistently even within one language (e.g. RU rows use
# both "..." and «...»), so nav-path detection must scan all of them, not just ASCII.
# (open, close); symmetric ASCII uses a dedicated char-class pattern to preserve behavior.
_NAV_QUOTE_PAIRS = [
    ('"', '"'),        # ASCII straight
    ('«', '»'),        # guillemets (Russian, French, ...)
    ('„', '“'),        # German low-high
    ('“', '”'),        # curly / CJK fullwidth
    ('「', '」'),       # Japanese corner brackets
    ('『', '』'),       # Japanese white corner brackets
]

# Bracket pairs that must never wrap a glossary term in ANY context.
_GLOSSARY_BRACKET_WRAP_PAIRS = (("[", "]"), ("「", "」"))

# Quotation-mark pairs that must not wrap a STANDALONE glossary term in title/heading/button
# copy only. Elsewhere these legitimately quote navigation paths or source text, so they are
# never treated as glossary wrappers outside the title/button context.
_TITLE_QUOTE_WRAP_PAIRS = (
    ('«', '»'),        # guillemets (Russian, French, ...)
    ('"', '"'),        # ASCII straight
    ('“', '”'),        # curly / CJK fullwidth double
    ('„', '“'),        # German low-high
    ('‘', '’'),        # curly single
)

# A glossary/brand *proper noun* token whose internal spacing must survive in every
# space-using language (e.g. "SmartThings", "Galaxy", "Bixby"). Used by the brand-
# concatenation lint that flags a dropped space such as
# "SmartThings Family Care" -> "SmartThingsFamily Care".
_BRAND_TOKEN_RE = re.compile(r'^[A-Z][A-Za-z0-9]*$')
_MIN_BRAND_TOKEN_LEN = 3

# ----------------- Case Hard-rule Applicable Languages -----------------
CASE_APPLICABLE_LANG_PREFIXES = {
    "English", "German", "French", "Spanish", "Portuguese",
    "Italian", "Dutch", "Swedish", "Polish", "Turkish",
    "Indonesian", "Vietnamese", "Russian"
}


def _is_case_sensitive_language(lang_name: str) -> bool:
    if not lang_name:
        return False
    return any(lang_name.startswith(pref) for pref in CASE_APPLICABLE_LANG_PREFIXES)


class GlossaryChecker:
    """글로서리 로드 + 매칭/케이싱/브래킷/브랜드 띄어쓰기 검사 상태와 로직을 소유한다."""

    def __init__(self, prompt_builder: PromptBuilder):
        self.prompt_builder = prompt_builder
        # Glossary Structure: { source_term: { 'targets': {code: term, ...}, 'rule': ... } }
        self.glossary = {}
        self.glossary_re = None  # Pre-compiled regex for fast lookup
        self.glossary_map = {}   # {lower_case_term: original_case_term}
        self.glossary_headers = []
        self.source_lang_code = None
        self._glossary_word_set = set()
        self.activation_manifest = OccurrenceActivationManifest()

    def load_activation_manifest(self, path: str | None) -> None:
        """Load occurrence activation separately from the glossary lexical data."""
        self.activation_manifest = OccurrenceActivationManifest.from_file(path)

    def _constraint_resolver(self) -> ConstraintResolver:
        return ConstraintResolver(
            glossary=self.glossary,
            get_relevant_terms=self._get_relevant_glossary_terms,
            get_target=self._get_target_val,
            is_deactivated=self.prompt_builder.is_glossary_deactivated,
            has_exempt_marker=self.prompt_builder.has_exempt_marker,
            get_context_mode=self.prompt_builder.get_glossary_context_mode,
            activation_manifest=self.activation_manifest,
        )

    def resolve_constraints(self, source_text: str, target_lang_code: str, *, row_key: str = "",
                            story: str | None = None, cell: str | None = None,
                            inside_navigation_path: bool = False):
        """Return the authoritative lexical/activation/bracket constraint card."""
        resolver = self._constraint_resolver()
        constraints = resolver.resolve(source_text, target_lang_code, row_key=row_key,
                                       story=story, cell=cell,
                                       inside_navigation_path=inside_navigation_path)
        return resolver.card(constraints)

    def validate_constraints(self, target_text: str, constraint_card: dict):
        """Validate a candidate without allowing the candidate to alter rules."""
        from translation_web_app.constraint_resolver import TermConstraint
        constraints = [TermConstraint(**term) for term in constraint_card.get("terms", [])]
        nav_spans = self._get_navigation_path_spans(target_text)
        return self._constraint_resolver().validate_target(target_text, constraints, navigation_spans=nav_spans)

    async def load_glossary_from_file(self, file_path: str, source_lang_code: str):
        """
        3행 구조의 용어집 CSV를 로드합니다.
        1행: 한국어, 영어_미국 등 (JSON 'code'와 매칭)
        2행: ko_KR, en_US 등
        3행: Lng 행
        """
        import os

        try:
            if not os.path.exists(file_path):
                return f"Glossary file not found: {file_path}"

            # 1. 헤더 3개 행을 먼저 읽어서 컬럼 인덱스 파악
            import pandas as pd
            df_headers = pd.read_csv(file_path, header=None, nrows=3, encoding='utf-8-sig').fillna("")

            num_cols = df_headers.shape[1]
            source_col_idx = -1
            rule_col_idx = -1
            # 소스 언어 컬럼 찾기 (1~3행 모두 검색)
            # Korean source: do NOT search for the Korean translation column here.
            # The key column (col 0) is always the primary identifier regardless of source language.
            # The Korean translation column is handled separately via kr_col_idx for double-registration.
            is_korean_source_pre = any(
                t in source_lang_code.lower() for t in ["korean", "한국어", "ko_kr", "ko-kr"]
            ) if source_lang_code else False

            search_terms = []
            if source_lang_code and not is_korean_source_pre:
                search_terms.append(source_lang_code.lower())
                lang_alias = {
                    "english": ["영어", "en_us", "en-us", "us", "en_gb", "en-gb", "uk", "en"],
                    "영어": ["english", "en_us", "en-us", "us", "en_gb", "en-gb", "uk", "en"]
                }
                if source_lang_code.lower() in lang_alias:
                    search_terms.extend(lang_alias[source_lang_code.lower()])

            if search_terms:
                for c in range(num_cols):
                    for r in range(3):
                        val = str(df_headers.iloc[r, c]).strip().lower()
                        if val and any(term == val or term in val or val in term for term in search_terms):
                            source_col_idx = c
                            break
                    if source_col_idx != -1: break

            # Korean source always falls back to col 0 (the Key/Lng identifier column).
            # This ensures "SmartThings" etc. in Korean source text can be matched by English key.
            if source_col_idx == -1:
                source_col_idx = 0
                if not is_korean_source_pre:
                    print(f"ℹ Glossary: '{source_lang_code}' 컬럼을 찾지 못해 0번 컬럼을 기본값으로 사용합니다.")

            # 설명/규칙/비고 컬럼 찾기
            for c in range(num_cols):
                for r in range(3):
                    val = str(df_headers.iloc[r, c]).strip().lower()
                    if any(kw in val for kw in ["설명", "규칙", "rule", "비고", "note", "remark", "desc"]):
                        rule_col_idx = c
                        break
                if rule_col_idx != -1: break

            if rule_col_idx == -1:
                print("⚠ Glossary: '규칙/비고' 컬럼을 찾지 못했습니다. 규칙이 적용되지 않을 수 있습니다.")

            # 2. 데이터 로드 (4행부터)
            df_data = pd.read_csv(file_path, header=None, skiprows=3, encoding='utf-8-sig').fillna("")

            self.source_lang_code = source_lang_code

            # 소스 언어가 한국어인 경우 한국어 컬럼도 찾아서 이중 키 등록에 사용
            is_korean_source = any(
                t in source_lang_code.lower() for t in ["korean", "한국어", "ko_kr", "ko-kr"]
            )
            kr_col_idx = -1
            if is_korean_source:
                for c in range(num_cols):
                    for r in range(3):
                        val = str(df_headers.iloc[r, c]).strip().lower()
                        if val and any(t in val or val in t for t in ["한국어", "ko_kr", "ko-kr", "korean"] if t):
                            kr_col_idx = c
                            break
                    if kr_col_idx != -1:
                        break

            count = 0
            for _, row in df_data.iterrows():
                source_term = str(row[source_col_idx]).strip()
                if not source_term or source_term.lower() == 'lng':
                    continue

                rule = str(row[rule_col_idx]).strip() if rule_col_idx != -1 else ""

                targets = {}
                for c in range(num_cols):
                    if c == source_col_idx or c == rule_col_idx:
                        continue

                    # 타겟 키는 1행(영어_미국 등) 또는 2행(en_US 등)에서 가져옴
                    header_key = str(df_headers.iloc[0, c]).strip() or str(df_headers.iloc[1, c]).strip()
                    if not header_key or header_key.lower() in ['key', 'lng']:
                        continue

                    val = str(row[c]).strip()
                    if val:
                        targets[header_key] = val
                        # 2행의 코드값도 키로 추가 (en_US 등)
                        code_key = str(df_headers.iloc[1, c]).strip()
                        if code_key and code_key != header_key:
                            targets[code_key] = val

                if targets:
                    # 영문 key 컬럼 값으로 등록 (항상)
                    self.glossary[source_term] = {"targets": targets, "rule": rule}
                    count += 1

                    # 🔑 소스 언어가 한국어면 한국어 컬럼 값도 소스 키로 이중 등록
                    # 예) 'SmartThings' 키와 함께 '스마트싱스'도 키로 등록 → 한국어 원문 검수 시 매칭 가능
                    if is_korean_source and kr_col_idx != -1:
                        kr_term = str(row[kr_col_idx]).strip()
                        if kr_term and kr_term.lower() not in ("lng", "") and kr_term != source_term:
                            if kr_term not in self.glossary:
                                self.glossary[kr_term] = {"targets": targets, "rule": rule}

            self._compile_glossary_re()
            kr_note = " (한국어 원문 키 이중 등록 적용)" if is_korean_source and kr_col_idx != -1 else ""
            return f"✓ 용어집 로드 성공: {len(self.glossary)}개 항목 (매칭 기준: {source_lang_code}){kr_note}"

        except Exception as e:
            return f"Glossary load failed: {str(e)}"

    def _get_target_val(self, targets: dict, lang_code: str) -> str | None:
        """Flexible lang-code → target value lookup: exact → case-insensitive → substring."""
        if not lang_code:
            return None
        v = targets.get(lang_code)
        if v:
            return v
        lc = lang_code.lower()
        for k, val in targets.items():
            if k and k.lower() == lc:
                return val
        for k, val in targets.items():
            kl = k.lower() if k else ""
            if kl and (lc in kl or kl in lc):
                return val
        return None

    def _extract_glossary_target_terms(self, glossary_context_or_terms) -> list[str]:
        if not glossary_context_or_terms:
            return []

        raw_terms = []
        if isinstance(glossary_context_or_terms, dict):
            raw_terms = list(glossary_context_or_terms.values())
        elif isinstance(glossary_context_or_terms, (list, tuple, set)):
            raw_terms = list(glossary_context_or_terms)
        else:
            raw_terms = [glossary_context_or_terms]

        terms = []
        for value in raw_terms:
            if not value:
                continue
            clean_value = re.sub(r'\(EXCEPTION:.*?\)', '', str(value)).strip()
            if clean_value:
                terms.append(clean_value)
        return sorted(set(terms), key=len, reverse=True)

    def _restore_glossary_target_casing(self, target_text: str, glossary_context_or_terms) -> str:
        if not target_text:
            return target_text

        restored = target_text
        for term in self._extract_glossary_target_terms(glossary_context_or_terms):
            pattern = re.escape(term)
            if term[0].isalnum():
                pattern = r'(?<![a-zA-Z0-9])' + pattern
            if term[-1].isalnum():
                pattern = pattern + r'(?![a-zA-Z0-9])'
            restored = re.sub(pattern, term, restored, flags=re.IGNORECASE)
        return restored

    def _compile_glossary_re(self):
        """
        용어집의 소스어들을 하나의 정규표현식으로 컴파일하여 검색 성능을 최적화합니다.
        길이가 긴 단어부터 매칭되도록 정렬하여 부분 일치 오류를 방지합니다.
        """
        if not self.glossary:
            self.glossary_re = None
            self._glossary_word_set = set()
            return

        # Every ASCII token that appears in a glossary key or target value, lowercased.
        # Used by the brand-concatenation lint to recognise legitimate single-token
        # camelCase brands (SmartThings, YouTube) so that a shorter brand token ("Smart")
        # matching their prefix is not mistaken for a dropped space.
        self._glossary_word_set = self._build_glossary_word_set()

        # 단어 길이가 긴 순서대로 정렬 (긴 단어가 우선 매칭되도록)
        sorted_terms = sorted(self.glossary.keys(), key=len, reverse=True)
        self.glossary_map = {t.lower(): t for t in sorted_terms}
        patterns = []

        for term in sorted_terms:
            escaped = re.escape(term)
            # 영어/숫자로 시작/종료되는 경우 단어 경계(\b)와 유사한 로직 적용
            # (단, 한국어 조사가 붙는 경우를 고려하여 앞뒤가 영문/숫자인 경우만 경계 체크)
            pattern = escaped
            if term[0].isalnum():
                pattern = r'(?<![a-zA-Z0-9])' + pattern
            if term[-1].isalnum():
                pattern = pattern + r'(?![a-zA-Z0-9])'
            patterns.append(f"({pattern})")

        # 모든 패턴을 OR(|)로 결합
        try:
            self.glossary_re = re.compile('|'.join(patterns), re.IGNORECASE)
        except Exception as e:
            print(f"Regex compilation failed: {e}")
            self.glossary_re = None

    def _get_relevant_glossary_terms(self, source_text: str):
        """
        원문에서 용어집에 포함된 단어들을 추출합니다.
        """
        if not self.glossary or not self.glossary_re or not source_text:
            return []

        found_terms = set()
        for match in self.glossary_re.finditer(source_text):
            # 매칭된 텍스트를 소문자로 변환하여 원래 용어집 키 찾기
            matched_text = match.group(0).lower()
            original_term = self.glossary_map.get(matched_text)
            if original_term:
                found_terms.add(original_term)
        return list(found_terms)

    def _build_glossary_lines_for_code(self, target_lang_code: str, source_text: str = None):
        if not self.glossary:
            return "용어집 없음"

        # 원문이 주어지면 관련 용어만 필터링, 아니면 전체 (하위 호환성)
        if source_text:
            relevant_terms = self._get_relevant_glossary_terms(source_text)
        else:
            relevant_terms = list(self.glossary.keys())

        if not relevant_terms:
            return "관련 용어 없음"

        out = []
        for term in relevant_terms:
            meta = self.glossary[term]
            tgt = self._get_target_val(meta["targets"], target_lang_code)
            if not tgt:
                continue
            rule = meta.get("rule")
            rule_info = f" (규칙: {rule})" if rule else ""
            out.append(f"- 원어: {term} → 대상어({target_lang_code}): {tgt}{rule_info}")

        return "\n".join(out) if out else f"용어집에 '{target_lang_code}' 타겟 항목 없음"

    def _get_glossary_context_as_dict(
        self,
        target_lang_code: str,
        source_text: str | None = None,
        skip_deactivated: bool = False,
        row_key: str = "",
    ):
        """
        용어집 데이터를 JSON/Dict 형태로 반환합니다. (프롬프트 최적화용)
        """
        if not self.glossary:
            return {}

        # 정규표현식이 컴파일되지 않은 경우 컴파일 시도
        if not self.glossary_re:
            self._compile_glossary_re()

        relevant_terms = self._get_relevant_glossary_terms(source_text) if source_text else list(self.glossary.keys())

        context = {}
        context_is_title_button = self.prompt_builder.should_skip_brackets(row_key)

        for term in relevant_terms:
            tgt = self._get_target_val(self.glossary[term]["targets"], target_lang_code)
            # `rule` keeps its spaces: the exempt markers below contain them.
            rule = self.glossary[term].get("rule", "").lower()

            if skip_deactivated and self.prompt_builder.is_glossary_deactivated(rule):
                continue

            if tgt:
                if context_is_title_button:
                    # System prompt carries the no-bracket instruction; keep value clean.
                    context[term] = tgt
                else:
                    exempt_markers = self.prompt_builder.get_exempt_markers()
                    if any(marker in rule for marker in exempt_markers):
                        context[term] = f"{tgt} (EXCEPTION: Do NOT wrap '{tgt}' in brackets)"
                    else:
                        context[term] = tgt
        return context

    def _precheck_glossary_mismatch(self, source_text: str, target_text: str, target_lang_code: str):
        if not self.glossary or not target_lang_code:
            return []

        relevant_terms = self._get_relevant_glossary_terms(source_text)
        if not relevant_terms:
            return []

        mismatches = []
        tgt_lower = target_text.lower()

        for s_term in relevant_terms:
            meta = self.glossary[s_term]

            if self.prompt_builder.is_glossary_deactivated(meta.get("rule", "")):
                continue

            t_term = self._get_target_val(meta["targets"], target_lang_code)
            if not t_term: continue

            if t_term.lower() not in tgt_lower:
                mismatches.append(f"[미적용] '{s_term}' → '{t_term}' 미적용")
        return mismatches

    def _check_glossary_casing(self, source_text: str, target_text: str, target_lang_code: str):
        if not self.glossary or not target_lang_code or not source_text or not target_text:
            return []

        relevant_terms = self._get_relevant_glossary_terms(source_text)
        if not relevant_terms:
            return []

        issues = []
        tgt_lower = target_text.lower()

        for s_term in relevant_terms:
            meta = self.glossary[s_term]

            if self.prompt_builder.is_glossary_deactivated(meta.get("rule", "")):
                continue

            t_term = self._get_target_val(meta["targets"], target_lang_code)
            if not t_term: continue

            if t_term.lower() in tgt_lower:
                idx = tgt_lower.find(t_term.lower())
                if idx == -1: continue
                actual = target_text[idx:idx + len(t_term)]
                if actual.lower() == t_term.lower() and actual != t_term:
                    issues.append(
                        f"[대소문자] 용어집 '{t_term}'의 표기가 '{actual}'로 사용됨 "
                        "(용어집 대소문자는 제목/문장형 대소문자 규칙보다 우선)"
                    )
        return issues

    def _get_navigation_path_spans(self, target_text: str, target_lang: str = "") -> list:
        """
        Deterministic disclaimer exception: quoted ranges containing '>' are treated
        as navigation paths (breadcrumbs). Scans every known quote-pair style
        (:data:`_NAV_QUOTE_PAIRS`) regardless of ``target_lang`` — real targets mix
        ASCII "...", guillemets «...», fullwidth “...”, and Japanese 「...」 per row.

        ``target_lang`` is retained for signature compatibility but no longer gates
        which quote styles are considered.
        """
        text = target_text or ""
        spans = []
        for open_q, close_q in _NAV_QUOTE_PAIRS:
            if open_q == close_q:
                pattern = re.escape(open_q) + r'[^' + re.escape(close_q) + r']*' + re.escape(close_q)
            else:
                pattern = re.escape(open_q) + r'.*?' + re.escape(close_q)
            for match in re.finditer(pattern, text):
                if ">" in match.group(0):
                    spans.append(match.span())
        return spans

    def _is_inside_span(self, start: int, end: int, spans: list[tuple[int, int]]) -> bool:
        return any(span_start <= start and end <= span_end for span_start, span_end in spans)

    def _check_glossary_brackets(self, source_text: str, target_text: str, target_lang_code: str, target_lang: str, row_key: str = ""):
        """Deterministic bracket audit — asserts only the high-confidence direction:
        a glossary term that is WRONGLY wrapped in brackets where policy forbids it
        (term-level 대괄호 제외, or inside a navigation path). See
        :meth:`PromptBuilder.resolve_glossary_bracket_policy`.

        It deliberately does NOT flag *missing* brackets: on real multilingual data
        that direction is unreliable (glossary terms that double as common nouns, e.g.
        Doorbell→Campainha in prose, and unquoted breadcrumbs), producing exactly the
        false positives seen in review. Term application is covered by
        :meth:`_precheck_glossary_mismatch`; missing-bracket nuance is left to the LLM audit.
        """
        if not self.glossary or not target_lang_code or not source_text or not target_text:
            return []

        relevant_terms = self._get_relevant_glossary_terms(source_text)
        if not relevant_terms:
            return []

        pb = self.prompt_builder
        brackets = pb.get_brackets(target_lang)
        context_mode = pb.get_glossary_context_mode(row_key)
        # Nav-path exception only applies to disclaimer rows (project rule). Quoted paths
        # drive the no_bracket POLICY, so keep them high precision (quoted spans only).
        nav_spans = self._get_navigation_path_spans(target_text, target_lang) if context_mode == "disclaimer" else []

        # A term counts as "wrongly wrapped" only when tightly wrapped in the universal
        # glossary bracket '[]' (never ambiguous with quotation). In the structural
        # title/button context — where quotation marks cannot legitimately surround a
        # standalone glossary term — also flag the language bracket and quotation-mark
        # wrappers (e.g. «Мой день»). Quotes are never checked outside title/button, where
        # they legitimately quote navigation paths or source text.
        wrong_wrap_pairs = [("[", "]")]
        if pb.should_skip_brackets(row_key):
            if (brackets[0], brackets[1]) != ("[", "]"):
                wrong_wrap_pairs.append((brackets[0], brackets[1]))
            wrong_wrap_pairs.extend(_TITLE_QUOTE_WRAP_PAIRS)

        def _wrapping_pair(idx: int, idx_end: int):
            for lb, rb in wrong_wrap_pairs:
                if idx > 0 and target_text[idx - 1] == lb and idx_end < len(target_text) and target_text[idx_end] == rb:
                    return lb, rb
            return None

        issues = []
        seen = set()  # dedupe by target value so mixed occurrences report at most once

        for s_term in relevant_terms:
            meta = self.glossary[s_term]
            rule = meta.get("rule", "").lower()
            if pb.is_glossary_deactivated(rule):
                continue

            target_val = self._get_target_val(meta["targets"], target_lang_code)
            if not target_val:
                continue
            clean_val = re.sub(r'\(.*?\)', '', target_val).strip()
            if not clean_val or clean_val in seen:
                continue

            # Resolve policy PER OCCURRENCE: the same term can be "no_bracket" inside a
            # navigation path and "wrap" outside it within one cell.
            for m in re.finditer(re.escape(clean_val), target_text, re.IGNORECASE):
                idx, idx_end = m.start(), m.end()
                inside_nav = bool(nav_spans) and self._is_inside_span(idx, idx_end, nav_spans)
                policy = pb.resolve_glossary_bracket_policy(
                    row_key=row_key, rule_text=rule, inside_nav_path=inside_nav
                )
                if policy == "no_bracket":
                    pair = _wrapping_pair(idx, idx_end)
                    if pair:
                        wl, wr = pair
                        seen.add(clean_val)
                        if inside_nav:
                            issues.append(f"[괄호 오류] '{clean_val}'는 내비게이션 경로 내부 용어이므로 감싸면 안 되나 '{wl}{wr}'로 감싸짐")
                        else:
                            issues.append(f"[괄호 오류] '{clean_val}'는 규칙상 감싸면 안 되나 '{wl}{wr}'로 감싸짐")
                        break

        return issues

    def _check_disclaimer_linebreak(self, source_text: str, target_text: str, row_key: str = "") -> list:
        """Disclaimer cells that break across multiple lines must start EVERY
        line — including the first — with '* ' (asterisk + space). Confirmed
        against production story data (KR/US/DE all apply it to every line).

        Only enforced when the source itself already uses the convention (2+
        non-empty lines, all '*'-prefixed), so disclaimers that aren't a
        bulleted list don't get false-flagged.
        """
        if not source_text or not target_text:
            return []
        if self.prompt_builder.get_glossary_context_mode(row_key) != "disclaimer":
            return []

        source_lines = [ln for ln in source_text.split("\n") if ln.strip()]
        if len(source_lines) < 2 or not all(ln.strip().startswith("*") for ln in source_lines):
            return []

        issues = []
        for i, ln in enumerate((tl for tl in target_text.split("\n") if tl.strip()), 1):
            if not re.match(r'^\*\s', ln.strip()):
                snippet = ln.strip()[:40]
                issues.append(f"[디스클레이머 서식] {i}번째 줄이 '* '로 시작하지 않습니다: '{snippet}'")
        return issues

    def _check_nav_path_bracket_leak(self, target_text: str, row_key: str = "") -> list:
        """Korean source wraps a navigation path as a single '[Settings > X > Y]'
        span, but every translation target must convert it to that locale's
        quotation marks (project rule, confirmed against production story
        data). A literal bracket-wrapped path surviving in the output means the
        model failed to convert Korean's bracket convention.
        """
        if not target_text:
            return []
        if self.prompt_builder.get_glossary_context_mode(row_key) != "disclaimer":
            return []

        issues = []
        for m in re.finditer(r'\[[^\[\]]*>[^\[\]]*\]', target_text):
            issues.append(f"[nav path 서식] 한국어 원문의 대괄호 경로 표기가 타겟 언어 따옴표로 변환되지 않았습니다: '{m.group(0)}'")
        return issues

    def _split_context_terms(self, glossary_context):
        """Return (all_target_terms, exempt_target_terms) from a glossary context.

        Exempt terms are those the context annotated with an ``(EXCEPTION: ...)``
        marker (i.e. 대괄호 제외). Both lists are cleaned of that annotation and
        sorted longest-first so nested terms unwrap correctly.
        """
        if isinstance(glossary_context, dict):
            raw_values = list(glossary_context.values())
        elif isinstance(glossary_context, (list, tuple, set)):
            raw_values = list(glossary_context)
        elif glossary_context:
            raw_values = [glossary_context]
        else:
            raw_values = []

        all_terms, exempt_terms = [], []
        for value in raw_values:
            if not value:
                continue
            raw = str(value)
            is_exempt = "EXCEPTION" in raw.upper()
            clean = re.sub(r'\(EXCEPTION:.*?\)', '', raw).strip()
            if not clean:
                continue
            all_terms.append(clean)
            if is_exempt:
                exempt_terms.append(clean)

        by_len = lambda seq: sorted(set(seq), key=len, reverse=True)
        return by_len(all_terms), by_len(exempt_terms)

    def _unwrap_glossary_brackets(self, text: str, terms, spans=None, pairs=_GLOSSARY_BRACKET_WRAP_PAIRS) -> str:
        """Remove wrapper pairs (``[term]`` / ``「term」`` by default) around glossary
        target terms.

        ``pairs`` selects which wrapper characters count: bracket pairs everywhere, plus
        quotation-mark pairs only in title/button context (see caller). If ``spans`` is
        given, only occurrences fully inside one of those spans are unwrapped
        (span-scoped); otherwise every occurrence is unwrapped. Matches are collected
        against the unchanged input and rebuilt in a single pass, so the stripping is not
        order/index dependent.
        """
        if not text or not terms:
            return text

        matches = []
        for term in terms:
            for left, right in pairs:
                pattern = re.escape(left) + r"\s*" + re.escape(term) + r"\s*" + re.escape(right)
                for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                    if spans is None or self._is_inside_span(m.start(), m.end(), spans):
                        matches.append((m.start(), m.end(), term))
        if not matches:
            return text

        matches.sort(key=lambda x: x[0])
        parts, last = [], 0
        for start, end, replacement in matches:
            if start < last:  # overlapping match (nested term) — skip
                continue
            parts.append(text[last:start])
            parts.append(replacement)
            last = end
        parts.append(text[last:])
        return "".join(parts)

    def _strip_glossary_brackets_by_policy(self, target_text: str, glossary_context, row_key: str = "") -> str:
        """
        Deterministic guardrail after LLM generation, policy-driven and span-aware.

        Prompt instructions are necessary but not sufficient. This removes glossary
        wrappers ONLY where :meth:`PromptBuilder.resolve_glossary_bracket_policy`
        says the term must not be wrapped — and never adds any:

        - title/button context      → unwrap every glossary term (whole cell), including
          quotation marks/guillemets (models wrap standalone feature names as «term»)
        - term-level 대괄호 제외      → unwrap brackets around that term (whole cell)
        - disclaimer navigation path → unwrap brackets ONLY inside the path span, so a
          generic term correctly bracketed outside the path — and the path's own quotation
          marks — are preserved (mixed cell)
        """
        if not target_text or not glossary_context:
            return target_text

        pb = self.prompt_builder
        context_mode = pb.get_glossary_context_mode(row_key)
        title_button = pb.should_skip_brackets(row_key)

        all_terms, exempt_terms = self._split_context_terms(glossary_context)

        cleaned = target_text
        if title_button:
            # Whole-cell unwrap of brackets AND quotation marks around standalone terms.
            if all_terms:
                cleaned = self._unwrap_glossary_brackets(
                    cleaned, all_terms, spans=None,
                    pairs=_GLOSSARY_BRACKET_WRAP_PAIRS + _TITLE_QUOTE_WRAP_PAIRS,
                )
        else:
            # 대괄호 제외 terms: unwrap brackets (never quotes) anywhere in the cell.
            if exempt_terms:
                cleaned = self._unwrap_glossary_brackets(cleaned, exempt_terms, spans=None)
            # Disclaimer nav path: unwrap brackets ONLY inside path spans; keep path quotes.
            if context_mode == "disclaimer":
                nav_spans = self._get_navigation_path_spans(cleaned)
                if nav_spans:
                    cleaned = self._unwrap_glossary_brackets(cleaned, all_terms, spans=nav_spans)

        return cleaned

    # Backward-compatible alias: the old name only handled title/button context,
    # which is now a subset of the policy-driven guardrail above.
    def _strip_title_button_glossary_brackets(self, target_text: str, glossary_context, row_key: str = "") -> str:
        return self._strip_glossary_brackets_by_policy(target_text, glossary_context, row_key)

    def _build_glossary_word_set(self) -> set:
        """Lowercased ASCII tokens across every glossary key and target value."""
        words: set[str] = set()
        for s_term, meta in (self.glossary or {}).items():
            values = list((meta.get("targets") or {}).values())
            values.append(s_term)
            for value in values:
                clean = re.sub(r'\(.*?\)', '', value or '').strip()
                for piece in re.split(r'[,/]', clean):
                    for token in piece.split():
                        if token.isascii():
                            words.add(token.lower())
        return words

    def _brand_like_target_terms(self, source_text: str) -> list[str]:
        """ASCII proper-noun tokens from the glossary terms relevant to ``source_text``.

        Brands keep their Latin spelling in every locale (SmartThings, Galaxy, Bixby),
        so they are the tokens whose internal spacing a translation must never drop.
        Multi-word brands are split into their tokens ("Galaxy Device" -> Galaxy,
        Device); only capitalized ASCII tokens survive, which excludes common-noun
        glossary targets (Doorbell->campainha) and non-Latin transliterations.
        """
        pb = self.prompt_builder
        tokens: set[str] = set()
        for s_term in (self._get_relevant_glossary_terms(source_text) or []):
            meta = self.glossary.get(s_term)
            if not meta:
                continue
            if pb.is_glossary_deactivated(meta.get("rule", "")):
                continue
            candidates = list(meta.get("targets", {}).values())
            candidates.append(s_term)
            for cand in candidates:
                clean = re.sub(r'\(.*?\)', '', cand or '').strip()
                for piece in re.split(r'[,/]', clean):
                    for token in piece.split():
                        if (
                            len(token) >= _MIN_BRAND_TOKEN_LEN
                            and token.isascii()
                            and _BRAND_TOKEN_RE.match(token)
                        ):
                            tokens.add(token)
        return sorted(tokens, key=len, reverse=True)

    def _check_brand_concatenation(self, source_text: str, target_text: str, target_lang_code: str, target_lang: str = "") -> list:
        """Flag a brand/glossary proper noun glued to an adjacent word with no space
        — the dropped-space signature ``SmartThingsFamily Care`` / ``GalaxyDevice``.

        High precision by construction, so no per-language allow-list is needed:
          * only ASCII proper-noun brand tokens are considered (common nouns excluded);
          * the brand must sit at a clean word boundary (never matched mid-word);
          * the glued neighbour must be an ASCII letter of the *anomalous* case — an
            UPPERCASE letter after the brand, or a lowercase word before it. Lowercase
            agglutinative suffixes (Turkish ``SmartThings'te``), Cyrillic case endings,
            and CJK characters are never ASCII+that-case, so they cannot trigger.
        """
        if not self.glossary or not source_text or not target_text:
            return []
        brand_tokens = self._brand_like_target_terms(source_text)
        if not brand_tokens:
            return []

        n = len(target_text)

        def _alnum(ch: str) -> bool:
            return ch.isalnum()

        def _fragment(i: int, j: int) -> str:
            left = i
            while left > 0 and _alnum(target_text[left - 1]):
                left -= 1
            right = j
            while right < n and _alnum(target_text[right]):
                right += 1
            return target_text[left:right]

        issues, seen = [], set()
        for token in brand_tokens:
            tl = len(token)
            start = 0
            while True:
                i = target_text.find(token, start)
                if i < 0:
                    break
                start = i + 1
                j = i + tl
                left_clean = (i == 0) or (not _alnum(target_text[i - 1]))
                right_clean = (j >= n) or (not _alnum(target_text[j]))

                after_glue = (
                    left_clean and j < n
                    and target_text[j].isascii() and target_text[j].isupper() and target_text[j].isalpha()
                )
                before_glue = (
                    right_clean and i > 0
                    and target_text[i - 1].isascii() and target_text[i - 1].islower()
                )
                if not (after_glue or before_glue):
                    continue

                frag = _fragment(i, j)
                # A merged run that is itself a known single-token brand (e.g. "SmartThings",
                # matched via its "Smart" prefix) is legitimate camelCase, not a lost space.
                if frag.lower() in getattr(self, "_glossary_word_set", set()):
                    continue
                if frag in seen:
                    continue
                seen.add(frag)
                if after_glue:
                    issues.append(
                        f"[공백 결합] 브랜드 '{token}' 뒤에 공백 없이 다음 단어가 붙었습니다: '{frag}' — 띄어쓰기 누락 의심"
                    )
                else:
                    issues.append(
                        f"[공백 결합] 브랜드 '{token}' 앞 단어가 공백 없이 붙었습니다: '{frag}' — 띄어쓰기 누락 의심"
                    )
        return issues

    def _analyze_sentence_case(self, target_text: str, target_lang: str, glossary_terms=None):
        if not target_text or not _is_case_sensitive_language(target_lang):
            return None, None

        text = target_text.strip()
        sentences = re.split(r'(?<=[\.!?])\s+', text)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return None, None

        report_lines = []
        fixed_sentences = []

        for idx, sent in enumerate(sentences, start=1):
            s = sent
            first_alpha_index = None
            for i, ch in enumerate(s):
                if ch.isalpha():
                    first_alpha_index = i
                    break

            if first_alpha_index is None:
                fixed_sentences.append(s)
                continue

            first_char = s[first_alpha_index]
            rest = s[first_alpha_index+1:]

            is_sentence_case = first_char.isupper() and rest == rest.lower()
            extra_caps = sum(1 for ch in rest if ch.isalpha() and ch.isupper())

            status = "문장형(첫 글자만 대문자)" if is_sentence_case else "문장형 아님"
            report_lines.append(f"- 문장 {idx}: {status}, 추가 대문자 수: {extra_caps}개")

            fixed_rest = rest.lower()
            fixed_sent = s[:first_alpha_index] + first_char.upper() + fixed_rest
            fixed_sent = self._restore_glossary_target_casing(fixed_sent, glossary_terms)
            fixed_sentences.append(fixed_sent)

        simple_fixed_text = " ".join(fixed_sentences)
        report = "\n".join(report_lines)
        if simple_fixed_text == target_text:
            simple_fixed_text = None
        return report, simple_fixed_text
