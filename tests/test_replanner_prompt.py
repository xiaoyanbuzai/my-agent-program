from web_app import _replanner_prompt


def test_replanner_prompt_requests_one_integrated_comparison():
    question = "比较两篇论文的训练方法"
    history = "论文 A 的结论。[来源: A p.1]\n论文 B 的结论。[来源: B p.2]"

    prompt = _replanner_prompt(question, history)

    assert question in prompt
    assert history in prompt
    assert "只生成一份完整、连贯的答复" in prompt
    assert "合并重复观点，只陈述一次" in prompt
    assert "共同点、关键差异" in prompt
    assert "不要把逐篇摘要拼接成答案" in prompt
