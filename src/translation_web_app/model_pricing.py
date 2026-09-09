"""모델 단가 단일 출처 (USD per 1M tokens, standard·paid tier 기준).

출처 — 2026-09-09 공식 페이지 확인:
  - Gemini: https://ai.google.dev/gemini-api/docs/pricing
  - OpenAI: https://developers.openai.com/api/docs/pricing

기준 정리:
  - standard(paid) tier 값이다. Batch/Flex(약 50%)·Priority 는 반영하지 않는다.
  - Gemini 출력 단가는 thinking 토큰을 포함한 값이다.
  - Gemini 3.6/3.7/3.8 Flash 는 2026-12-31 까지 도입가가 적용되고 2027-01-01 부터
    표준가(2배)로 오른다. 날짜에 따라 자동 전환하므로 연초에 추정치가 조용히
    절반으로 어긋나지 않는다.
"""

from datetime import date

# 도입가 적용 마지막 날 (이 날짜까지 INTRO, 다음 날부터 STANDARD)
INTRO_UNTIL = date(2026, 12, 31)

# model -> (input, output) USD per 1M tokens
_STANDARD: dict[str, tuple[float, float]] = {
    # Gemini — 2027-01-01 이후 적용가
    "gemini-3.8-flash": (1.50, 7.50),
    "gemini-3.7-flash": (1.50, 7.50),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash": (1.50, 9.00),  # 도입가 없음
    # Pro 는 프롬프트 길이로 단가가 갈린다. 앱은 셀 단위 번역이라 200k 근처도 가지 않으므로
    # <=200k 구간을 쓴다. >200k 프롬프트를 다루게 되면 (4.00, 18.00) 로 분기해야 한다.
    "gemini-3.1-pro-preview": (2.00, 12.00),
    # OpenAI
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.2": (1.75, 14.00),
}

# INTRO_UNTIL 까지만 적용되는 도입가 (없는 모델은 _STANDARD 를 그대로 쓴다)
_INTRO: dict[str, tuple[float, float]] = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
}


def price_per_1m(model: str, on: date | None = None) -> tuple[float, float]:
    """해당 시점에 유효한 (input, output) USD per 1M tokens 를 돌려준다."""
    if model not in _STANDARD:
        raise KeyError(
            f"단가 미등록 모델: {model!r} — model_pricing.py 에 공식 페이지 값을 추가하세요. "
            f"등록된 모델: {', '.join(sorted(_STANDARD))}"
        )
    on = on or date.today()
    if on <= INTRO_UNTIL and model in _INTRO:
        return _INTRO[model]
    return _STANDARD[model]


def pricing_table(models: list[str] | None = None, on: date | None = None) -> dict[str, dict[str, float]]:
    """{model: {"input": per_token, "output": per_token}} — 기존 PRICING dict 형태.

    단가는 토큰당 값이라 `tokens * table[m]["input"]` 으로 바로 비용이 나온다.
    """
    names = models if models is not None else list(_STANDARD)
    table = {}
    for m in names:
        inp, out = price_per_1m(m, on)
        table[m] = {"input": inp / 1_000_000, "output": out / 1_000_000}
    return table


def pricing_note(on: date | None = None) -> str:
    """리포트 하단에 붙일 단가 기준 한 줄."""
    on = on or date.today()
    phase = f"도입가 적용중(~{INTRO_UNTIL})" if on <= INTRO_UNTIL else "표준가"
    return (f"※ 단가 기준: 공식 페이지(2026-09-09 확인), standard tier, {phase}. "
            f"Gemini 3.6~3.8 Flash 는 {INTRO_UNTIL.year + 1}-01-01 부터 2배로 오릅니다.")
