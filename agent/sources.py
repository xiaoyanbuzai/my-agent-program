"""取证来源词汇表：来源编号、标签、线索词、实体正则。"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# 意图层：先判「通用直答」还是「外部检索」，再决定「查哪个源」
#
# 路由回答的是「怎么答」(多步还是直答)，意图回答的是「要不要外部取证」。
# 两者必须分开判，否则「什么是 Transformer」会被绑上知识库检索，
# 而「取 PMID 42260443 的摘要」又可能被当成通用知识直接编。
#
# 判定顺序(越靠前越确定)：
#   1. 硬规则：空问题 / 用户显式「不要检索」/ 寒暄 / Hermes / 论文生成
#   2. 实体锚点：PMID、DOI、arXiv 编号、URL、rsID、HGVS 变异、疾病编码
#   3. 来源线索：PubMed、arXiv、联网新闻、本地文档指代、基因/化合物/引用
#   4. 兜底：能归入通用概念解释 → direct(不绑任何工具)；否则 → 知识库检索
# ---------------------------------------------------------------------------

_SOURCE_KB = "kb"
_SOURCE_PUBMED = "pubmed"
_SOURCE_ARXIV = "arxiv"
_SOURCE_WEB = "web"
_SOURCE_URL = "url"
_SOURCE_GENE = "gene"
_SOURCE_COMPOUND = "compound"
_SOURCE_ICD = "icd"
_SOURCE_CITATION = "citation"
_SOURCE_IMAGE = "image"
_SOURCE_HERMES = "hermes"
_SOURCE_DOCX = "docx"

_SOURCE_LABELS = {
    _SOURCE_KB: "知识库",
    _SOURCE_PUBMED: "PubMed",
    _SOURCE_ARXIV: "arXiv/论文",
    _SOURCE_WEB: "联网资讯",
    _SOURCE_URL: "网页",
    _SOURCE_GENE: "基因/变异",
    _SOURCE_COMPOUND: "化合物/药物",
    _SOURCE_ICD: "疾病编码",
    _SOURCE_CITATION: "引用网络",
    _SOURCE_IMAGE: "医学图像",
    _SOURCE_HERMES: "Hermes 平台",
    _SOURCE_DOCX: "Word 文档",
}

# 真正去查外部数据的来源；知识库属于「私有检索」，单独判断
_EXTERNAL_SOURCES = frozenset(
    {
        _SOURCE_PUBMED,
        _SOURCE_ARXIV,
        _SOURCE_WEB,
        _SOURCE_URL,
        _SOURCE_GENE,
        _SOURCE_COMPOUND,
        _SOURCE_ICD,
        _SOURCE_CITATION,
        _SOURCE_IMAGE,
        _SOURCE_HERMES,
    }
)

_MODE_DIRECT = "direct"
_MODE_RETRIEVAL = "retrieval"

# ---- 实体锚点：出现即说明「这是一次针对具体对象的检索」 --------------------
_PMID_PATTERN = re.compile(r"(?<!\d)(\d{6,9})(?!\d)")
_DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[^\s,;，；)）\]]+", re.I)
_ARXIV_ID_PATTERN = re.compile(r"\barxiv\s*[:：]?\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_RSID_PATTERN = re.compile(r"\brs\d{3,9}\b", re.I)
_HGVS_PATTERN = re.compile(r"\b[cp]\.(?:\d+[acgt]>[acgt]|[a-z]{3}\d+[a-z]{3})", re.I)
_NCT_PATTERN = re.compile(r"\bnct\d{8}\b", re.I)
_ICD_CODE_PATTERN = re.compile(r"\b[a-tv-z]\d{2}\.\d{1,4}\b", re.I)
_URL_PATTERN = re.compile(r"https?://\S+")
# 全大写单词(TP53/KRAS/EGFR/ALK…)再与已知基因符号表取交集，避免误判普通缩写
_GENE_SYMBOL_TOKENS = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")

# 常见抑癌/致癌基因：补上「KRAS」这种不带数字的符号，避免只认 TP53
_KNOWN_GENE_SYMBOLS = frozenset(
    {
        "TP53", "KRAS", "NRAS", "HRAS", "EGFR", "ERBB2", "HER2", "ALK", "ROS1",
        "BRAF", "MET", "RET", "PTEN", "RB1", "APC", "MYC", "PIK3CA", "BRCA1",
        "BRCA2", "ATM", "CHEK2", "PALB2", "CDKN2A", "IDH1", "IDH2", "VHL",
        "NF1", "NF2", "SMAD4", "STK11", "MLH1", "MSH2", "MSH6", "PMS2",
        "JAK2", "FLT3", "KIT", "PDGFRA", "ESR1", "AR", "MKI67", "TMB", "MSI",
    }
)

# ---- 来源线索 -------------------------------------------------------------
_KB_REFERENCE_CUES = (
    "知识库",
    "我的文档",
    "我上传",
    "上传的",
    "文档中",
    "资料里",
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
_PUBMED_SOURCE_CUES = (
    "pubmed",
    "ncbi",
    "pmc",
    "文献综述",
    "综述",
    "review",
    "系统评价",
    "meta 分析",
    "meta-analysis",
    "研究进展",
)
_ARXIV_SOURCE_CUES = ("arxiv", "预印本", "preprint")
_WEB_SOURCE_CUES = (
    "联网",
    "全网",
    "网页搜索",
    "新闻",
    "最新消息",
    "最新资讯",
    "今天发生",
    "最近发生",
    "近期动态",
)
_GENE_CUES = (
    "基因",
    "突变",
    "位点",
    "变异",
    "variant",
    "mutation",
    "germline",
    "somatic",
    "表达水平",
    "蛋白",
    "通路",
    "clinvar",
    "hgvs",
)
_COMPOUND_CUES = (
    "化合物",
    "小分子",
    "药物",
    "靶点",
    "抑制剂",
    "新药",
    "靶向药",
    "compound",
    "drug",
    "inhibitor",
    "ic50",
    "分子对接",
)
_ICD_CUES = ("icd", "疾病编码", "icd-10", "icd-11", "mesh")
_CITATION_CUES = (
    "引用",
    "被引",
    "引文",
    "citing",
    "citation",
    "相似文献",
    "相关文献",
    "参考文献",
)
_IMAGE_CUES = ("影像", "放射", "图片", "figure", "image", "病理图", "ct 图")
_PROGNOSIS_CUES = (
    "预后",
    "生存",
    "总生存",
    "无进展",
    "hazard",
    "风险比",
    "hr",
    "os",
    "pfs",
    "随访",
    "队列",
)
_FRESHNESS_CUES = ("最新", "最近", "今年", "去年", "进展", "趋势", "前沿", "state of the art")
