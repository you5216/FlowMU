                                                                                     
import argparse
import csv
import json
from pathlib import Path

from .mia_stats import compare, norm


def answer_coverage(context, answer):
    reference = set(norm(answer).split())
    tokens = set(norm(context).split())
    return len(reference & tokens) / len(reference) if reference else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Summarize fixed mixed-source MIA component scores.')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--nll-summary', type=Path)
    args = parser.parse_args(argv)
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = ['Vanilla', 'PURE', 'SBU', 'SF-NL', 'FlowMU']
    metrics = []
    nll_path = args.nll_summary or input_dir / 'parametric_nll/summary.json'
    if args.nll_summary is not None and not nll_path.is_file():
        raise FileNotFoundError(nll_path)
    nll_results = json.loads(nll_path.read_text(encoding='utf-8'))['results'] if nll_path.exists() else []
    for method in methods:
        with (input_dir / (method + '_results.csv')).open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 608
        for row in rows:
            row['rag_answer_coverage'] = answer_coverage(row.get('rag_context', ''), row['reference_answer'])
            row['memory_answer_coverage'] = answer_coverage(row.get('memory_context', ''), row['reference_answer'])
            row['rouge_l'] = float(row['rouge_l'])
        for component, feature in [('RAG', 'rag_answer_coverage'), ('Memory', 'memory_answer_coverage'), ('Agent', 'rouge_l')]:
            for pos in ['forget', 'retain']:
                metrics.append(dict(method=method, component=component, **compare(rows, pos, 'test', feature)))
    protocol = dict(status='post-hoc mixed-source component diagnostics',
                    rag_memory_score='fraction of unique normalized reference-answer words in retrieved context',
                    agent_score='ROUGE-L of final answer with reference',
                    threat_model='reference-aware context inspection for RAG/Memory; reference-aware output inspection for Agent',
                    membership='native-SFT labels reused; graph insertion membership not independently audited',
                    llm='parameter-only negative answer NLL on mixed set' if nll_results else 'not measured',
                    limitations=['Different components use different fixed scores; not directly comparable attack strengths.',
                                 'Empty/disabled RAG for SBU is non-applicable, not evidence of privacy.',
                                 'Synthetic vs auxiliary cancellation and starting-model confounding remain.',
                                 'Row bootstrap is exploratory; no author-cluster correction.'])
    for row in nll_results:
        if row['negative_split'] == 'test' and row['feature'] == 'neg_nll':
            metric = {k: v for k, v in row.items() if k != 'average_precision'}
            metrics.append(dict(metric, component='LLM'))
    (output_dir / 'component_summary.json').write_text(json.dumps(dict(protocol=protocol, results=metrics), indent=2), encoding='utf-8')
    with (output_dir / 'component_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    lines = ['# 两张结果表', '', '固定混合集：Forget=200、Retain=200、Non-member=208（Synthetic=100、Auxiliary=108）。', '',
             '## 表一：完整 Agent 输出攻击', '', '| 方法 | Forget vs Test AUROC | Retain vs Test AUROC |', '|---|---:|---:|']
    def value(method, component, positive):
        return next(r['auroc'] for r in metrics if r['method'] == method and r['component'] == component and r['positive_split'] == positive)
    for method in methods:
        lines.append(f"| {method} | {value(method,'Agent','forget'):.4f} | {value(method,'Agent','retain'):.4f} |")
    lines += ['', '## 表二：组件级攻击', '', '| 方法 | RAG F-AUC | RAG R-AUC | Memory F-AUC | Memory R-AUC | LLM F-AUC | LLM R-AUC |', '|---|---:|---:|---:|---:|---:|---:|']
    for method in methods:
        rags = ['不适用', '不适用'] if method == 'SBU' else [f"{value(method,'RAG',s):.4f}" for s in ['forget','retain']]
        memory = [f"{value(method,'Memory',s):.4f}" for s in ['forget','retain']]
        llm = []
        for pos in ['forget', 'retain']:
            metric = next((r for r in nll_results if r['method'] == method and r['positive_split'] == pos and r['negative_split'] == 'test' and r['feature'] == 'neg_nll'), None)
            llm.append(f"{metric['auroc']:.4f}" if metric else '未测')
        lines.append(f"| {method} | {rags[0]} | {rags[1]} | {memory[0]} | {memory[1]} | {llm[0]} | {llm[1]} |")
    lines += ['', 'RAG/Memory 分数是检索上下文对参考答案词的覆盖率；Agent 分数是输出 ROUGE-L。它们是固定的参考答案可见攻击诊断，不是训练好的组件攻击器。',
              'SBU 当前协议关闭 RAG。纯 LLM 使用同一混合集的无检索 direct-QA 负 answer NLL，SF-NL 不包含防护提示，因此与 Vanilla 相同。校准 NLL 另存于 parametric_nll/summary.json，不择优替换原始攻击。',
              '标签来自 native-SFT 成员身份，尚未为每种知识库独立核验插入成员身份。因此组件表目前应称为上下文泄露诊断，不是已完成的严格 component-membership 主实验。',
              '混合来源可能抵消信号；FlowMU 训练起点与基线不同。这两张表均为探索性结果，不能证明公平比较下成员泄露消失。']
    (output_dir / '完整结果.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
