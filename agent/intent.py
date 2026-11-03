"""意图边界：区分「通用直答」与「外部检索」，并点名取证来源。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.signals import (
    _PURE_GREETING_PATTERN,
    _ROUTE_DIRECT_MARKERS,
    _is_paper_request,
    _token_hit,
)

from agent.sources import _ARXIV_ID_PATTERN, _ARXIV_SOURCE_CUES, _CITATION_CUES, _COMPOUND_CUES, _DOI_PATTERN, _FRESHNESS_CUES, _GENE_CUES, _GENE_SYMBOL_TOKENS, _HGVS_PATTERN, _ICD_CODE_PATTERN, _ICD_CUES, _IMAGE_CUES, _KB_REFERENCE_CUES, _KNOWN_GENE_SYMBOLS, _MODE_DIRECT, _MODE_RETRIEVAL, _NCT_PATTERN, _PMID_PATTERN, _PROGNOSIS_CUES, _PUBMED_SOURCE_CUES, _RSID_PATTERN, _SOURCE_ARXIV, _SOURCE_CITATION, _SOURCE_COMPOUND, _SOURCE_DOCX, _SOURCE_GENE, _SOURCE_HERMES, _SOURCE_ICD, _SOURCE_IMAGE, _SOURCE_KB, _SOURCE_LABELS, _SOURCE_PUBMED, _SOURCE_URL, _SOURCE_WEB, _URL_PATTERN, _WEB_SOURCE_CUES


def _is_general_knowledge_request(text: str) -> bool:
    """识别明确的概念解释请求，避免为通用知识额外检索私有知识库。"""
    lowered = (text or "").lower()
    if _explicitly_references_knowledge_base(lowered):
        return False

    concept_patterns = (
        r"什么是|何为|定义(?:是什么)?|概念(?:是什么)?",
        r"核心思想|基本原理|工作原理|怎么(?:计算|理解|工作)|如何(?:计算|理解|工作)",
        r"有什么区别|区别(?:是什么|在哪)?|差异(?:是什么|在哪)?",
        r"\bwhat is\b|\bmeaning of\b|\bhow does\b|\bhow to\b|\bdifference between\b",
        r"\bcore idea\b|\bprinciple of\b",
    )
    return any(re.search(pattern, lowered) for pattern in concept_patterns)


def _explicitly_references_knowledge_base(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        term in lowered
        for term in (
            "知识库",
            "文档",
            "资料",
            "上传",
            "这篇论文",
            "该论文",
            "本文",
            "文中",
            "论文中",
            "文献中",
            "根据论文",
            "根据文献",
            "根据文章",
            "in the paper",
            "in this article",
            "my document",
            "uploaded",
        )
    )


@dataclass(frozen=True)
class IntentDecision:
    """一次意图判定的完整依据，可直接用于日志与前端提示。"""

    mode: str
    sources: tuple[str, ...]
    confidence: float
    reasons: tuple[str, ...]
    entities: tuple[str, ...] = ()
    entity_kinds: frozenset[str] = frozenset()

    @property
    def needs_retrieval(self) -> bool:
        return self.mode == _MODE_RETRIEVAL

    def source_text(self) -> str:
        if not self.sources:
            return "无"
        return "+".join(_SOURCE_LABELS.get(item, item) for item in self.sources)

    def reason_text(self) -> str:
        return " + ".join(self.reasons) if self.reasons else "无"


def _classify_intent(question: str) -> IntentDecision:
    """判定本轮要不要外部取证、取证目标是谁；纯函数，不读配置、不调模型。"""
    raw = " ".join(str(question or "").split())
    if not raw:
        return IntentDecision(_MODE_DIRECT, (), 1.0, ("空问题",))
    lowered = raw.lower()

    def hit(*cues: str) -> bool:
        return any(_token_hit(cue, lowered) for cue in cues)

    sources: list[str] = []
    reasons: list[str] = []
    labels: list[str] = []
    kinds: set[str] = set()

    def add(source: str, reason: str) -> None:
        if source not in sources:
            sources.append(source)
            reasons.append(reason)

    pmids = _PMID_PATTERN.findall(raw)
    dois = _DOI_PATTERN.findall(raw)
    arxiv_ids = _ARXIV_ID_PATTERN.findall(raw)
    urls = _URL_PATTERN.findall(raw)
    rsids = _RSID_PATTERN.findall(raw)
    hgvs = _HGVS_PATTERN.findall(raw)
    icd_codes = _ICD_CODE_PATTERN.findall(raw)
    nct_ids = _NCT_PATTERN.findall(raw)
    genes = [
        symbol
        for symbol in _GENE_SYMBOL_TOKENS.findall(raw)
        if symbol.upper() in _KNOWN_GENE_SYMBOLS
    ]

    for values, kind, prefix in (
        (pmids, "pmid", "PMID"),
        (dois, "doi", "DOI"),
        (arxiv_ids, "arxiv_id", "arXiv"),
        (urls, "url", "URL"),
        (rsids, "rsid", "rsID"),
        (hgvs, "hgvs", "变异"),
        (icd_codes, "icd", "ICD"),
        (nct_ids, "nct", "NCT"),
    ):
        if values:
            kinds.add(kind)
            labels.append(f"{prefix}×{len(values)}")
    if genes:
        kinds.add("gene_symbol")
        labels.append("基因符号:" + "/".join(sorted({item.upper() for item in genes})[:3]))

    # 1. 硬规则
    if _PURE_GREETING_PATTERN.fullmatch(raw):
        return IntentDecision(_MODE_DIRECT, (), 1.0, ("寒暄/自我介绍",), tuple(labels))
    if any(_token_hit(term, lowered) for term in _ROUTE_DIRECT_MARKERS):
        return IntentDecision(_MODE_DIRECT, (), 0.95, ("用户要求直接作答",), tuple(labels))
    if "hermes" in lowered or hit(
        "平台频道", "平台会话", "会话列表", "消息记录", "权限审批", "权限列表", "待处理权限"
    ):
        return IntentDecision(
            _MODE_RETRIEVAL, (_SOURCE_HERMES,), 0.95, ("Hermes 平台查询",), tuple(labels)
        )
    if _is_paper_request(raw):
        return IntentDecision(
            _MODE_RETRIEVAL,
            (_SOURCE_KB, _SOURCE_DOCX),
            0.95,
            ("论文文档生成",),
            tuple(labels),
            frozenset({"docx"}),
        )

    # 2. 实体锚点
    if urls:
        add(_SOURCE_URL, f"含 URL×{len(urls)}")
    if pmids or dois:
        add(_SOURCE_PUBMED, "含 PMID/DOI" if pmids else "含 DOI")
    if arxiv_ids:
        add(_SOURCE_ARXIV, "含 arXiv 编号")
    if nct_ids:
        add(_SOURCE_PUBMED, "含 NCT 临床试验编号")

    # 3. 来源线索
    search_hinted = hit(
        *_PUBMED_SOURCE_CUES, *_ARXIV_SOURCE_CUES, *_WEB_SOURCE_CUES
    )
    if hit(*_PUBMED_SOURCE_CUES):
        add(_SOURCE_PUBMED, "PubMed/综述线索")
    if hit(*_ARXIV_SOURCE_CUES):
        add(_SOURCE_ARXIV, "arXiv/预印本线索")
    if hit(*_WEB_SOURCE_CUES):
        add(_SOURCE_WEB, "联网/新闻线索")
    if hit(*_KB_REFERENCE_CUES):
        add(_SOURCE_KB, "指向本地文档")
    if rsids or hgvs or hit(*_GENE_CUES):
        add(_SOURCE_GENE, "基因/变异线索")
    elif genes and not search_hinted:
        # 只写了基因符号(如 EGFR)时也算生物医学检索；但如果已经明确说
        # “PubMed 最新综述”，就没必要再把基因工具塞进这一轮
        add(_SOURCE_GENE, "基因符号")
    if hit(*_COMPOUND_CUES):
        add(_SOURCE_COMPOUND, "化合物/药物线索")
    if hit(*_ICD_CUES) or (icd_codes and hit("编码", "诊断")):
        add(_SOURCE_ICD, "疾病编码线索")
    if hit(*_CITATION_CUES):
        add(_SOURCE_CITATION, "引用/被引线索")
    if hit(*_IMAGE_CUES):
        add(_SOURCE_IMAGE, "医学图像线索")
    if hit(*_PROGNOSIS_CUES):
        add(_SOURCE_PUBMED, "预后/生存线索")
    if hit(*_FRESHNESS_CUES) and not sources and _is_general_knowledge_request(raw):
        # “最新进展”但没有任何领域/来源线索时，仍按知识库问答处理，避免乱联网
        pass

    # 4. 兜底与补全
    explicit_search = bool(
        urls or pmids or dois or arxiv_ids or nct_ids
    ) or search_hinted
    if not sources:
        if _is_general_knowledge_request(raw) and not kinds:
            return IntentDecision(
                _MODE_DIRECT, (), 0.8, ("通用知识/概念解释",), tuple(labels)
            )
        return IntentDecision(
            _MODE_RETRIEVAL,
            (_SOURCE_KB,),
            0.55,
            ("无明显外部线索,先查知识库",),
            tuple(labels),
        )

    if not explicit_search and _SOURCE_KB not in sources:
        # 基因/化合物/编码这类“本地资料也可能有”的问题，同时带上知识库
        add(_SOURCE_KB, "本地资料补充")

    confidence = 0.95 if (kinds or explicit_search) else 0.7
    return IntentDecision(
        _MODE_RETRIEVAL,
        tuple(sources),
        confidence,
        tuple(reasons),
        tuple(labels),
        frozenset(kinds),
    )
