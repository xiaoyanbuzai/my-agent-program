"""最终答案硬化测试：数据优先 + 确定性剔除免责声明/学术废话。

提示词约束(【最终输出硬约束】)只能降低概率，真正兜底的是 _strip_answer_filler：
- 整行免责声明/套话直接丢弃；
- 带硬数据(数字/PMID/CI)的行只去掉开头套话，绝不整行删；
- 「免责声明」小节连同其后的内容一并截断。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from web_app import (
    _ANSWER_CONTRACT,
    _replanner_prompt,
    _sse,
    _stream_answer,
    _strip_answer_filler,
    _system_prompt,
)


def _frame(text: str) -> str:
    return f"data: {json.dumps({'type': 'done', 'content': text}, ensure_ascii=False)}"


# ------------------------------------------------------------------ 过滤

def test_disclaimer_line_is_dropped():
    text = "TP53 突变与较差预后相关（HR 1.8, 95%CI 1.2-2.7）。\n以上内容仅供参考，不构成医疗建议。"

    cleaned, dropped = _strip_answer_filler(text)

    assert "仅供参考" not in cleaned
    assert "HR 1.8" in cleaned
    assert dropped == 1


def test_academic_filler_line_is_dropped():
    text = "HR 1.8（95%CI 1.2-2.7），n=1234。\n综上所述，该方向具有重要意义。"

    cleaned, dropped = _strip_answer_filler(text)

    assert "综上所述" not in cleaned
    assert "HR 1.8" in cleaned
    assert dropped == 1


def test_data_line_keeps_data_but_loses_leading_filler():
    text = "综上所述，TP53 突变组中位 OS 为 18.2 个月（n=412）。"

    cleaned, _ = _strip_answer_filler(text)

    assert cleaned.startswith("TP53 突变组")
    assert "18.2" in cleaned


def test_disclaimer_section_is_truncated():
    text = (
        "结论：KRAS G12C 抑制剂 ORR 37%（n=126）。\n\n"
        "## 免责声明\n"
        "本文仅供学术交流，不能替代医生诊断。\n"
        "请咨询专业医师。"
    )

    cleaned, _ = _strip_answer_filler(text)

    assert "KRAS G12C" in cleaned
    assert "免责声明" not in cleaned
    assert "请咨询" not in cleaned


def test_retrieval_process_line_is_dropped():
    text = "根据检索结果，TP53 突变频率约 50%。"

    cleaned, _ = _strip_answer_filler(text)

    assert cleaned.startswith("TP53 突变频率")
    assert "根据检索结果" not in cleaned


def test_inline_disclaimer_is_removed():
    text = "结论：HR 1.8（仅供参考）。"

    cleaned, _ = _strip_answer_filler(text)

    assert "仅供参考" not in cleaned
    assert "HR 1.8" in cleaned


def test_clean_answer_is_unchanged():
    text = "TP53 突变型肺癌 5 年生存率 18%，野生型 32%（PMID 42260443）。"

    cleaned, dropped = _strip_answer_filler(text)

    assert cleaned == text
    assert dropped == 0


def test_repeated_blank_lines_are_collapsed():
    text = "结论 A。\n\n\n\n结论 B。"

    cleaned, _ = _strip_answer_filler(text)

    assert cleaned == "结论 A。\n\n结论 B。"


def test_empty_answer_stays_empty():
    assert _strip_answer_filler("") == ("", 0)
    assert _strip_answer_filler("   ") == ("", 0)


# ------------------------------------------------------------------ 契约

def test_answer_contract_forbids_disclaimers_and_filler():
    assert _ANSWER_CONTRACT in _system_prompt()
    assert _ANSWER_CONTRACT in _replanner_prompt("问题", "历史")
    assert "仅供参考" in _ANSWER_CONTRACT
    assert "综上所述" in _ANSWER_CONTRACT
    assert "第一行直接给结论" in _ANSWER_CONTRACT


def test_direct_mode_prompt_marks_general_answer():
    assert "通用直答" in _system_prompt(direct=True)
    assert "通用直答" not in _system_prompt()
    assert _system_prompt() == _system_prompt(), "提示词必须稳定以命中缓存"


# ------------------------------------------------------------------ 流式

class _GraphWithFiller:
    def stream(self, payload, config, stream_mode):
        yield SimpleNamespace(
            content="结论：HR 1.8。\n以上仅供参考，不构成医疗建议。"
        ), {"langgraph_node": "chatbot"}


def _events(chunks):
    return [json.loads(chunk[6:].strip()) for chunk in chunks]


def test_stream_done_event_is_cleaned():
    events = _events(_stream_answer(_GraphWithFiller(), "问题", {}))
    done = [event for event in events if event["type"] == "done"]

    assert done == [{"type": "done", "content": "结论：HR 1.8。"}]


def test_concise_can_be_disabled(monkeypatch):
    settings = SimpleNamespace(answer_concise=False, router_explain=False)
    monkeypatch.setattr("web_app.get_settings", lambda: settings)

    events = _events(_stream_answer(_GraphWithFiller(), "问题", {}))
    done = [event for event in events if event["type"] == "done"]

    assert "仅供参考" in done[0]["content"]


def test_sse_payload_still_utf8():
    payload = _sse({"type": "token", "content": "中文"})

    assert payload.startswith("data: ")
    assert "\\u" not in payload
