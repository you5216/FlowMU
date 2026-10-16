                                                                                     
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from .mia_stats import compare, rouge_l

def load(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len({r['question'] for r in rows}) == len(rows)
    return {r['question']: r for r in rows}

def main(argv=None):
    parser = argparse.ArgumentParser(description='Build the fixed mixed-source MIA evaluation.')
    parser.add_argument('--synthetic-dir', type=Path, required=True)
    parser.add_argument('--auxiliary-dir', type=Path, required=True)
    parser.add_argument('--tofu-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    new = args.synthetic_dir
    old = args.auxiliary_dir
    data = args.tofu_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    synthetic = load(new / 'test_synthetic.jsonl')
    assert len(synthetic) == 200
    indices = sorted(i * 20 + j for i in range(10) for j in rng.sample(range(20), 10))
    chosen = [(synthetic[i], 'synthetic', i) for i in indices]
    auxiliary = load(old / 'test_auxiliary.jsonl')
    source_questions = {s: {r['question'] for r in load(data / (s + '.json'))} for s in ['real_authors', 'world_facts']}
    for source, count in [('real_authors', 50), ('world_facts', 58)]:
        pool = [i for i, r in enumerate(auxiliary) if r['question'] in source_questions[source]]
        for i in sorted(rng.sample(pool, count)):
            chosen.append((auxiliary[i], source, i))
    assert len(chosen) == 208 and len({r['question'] for r, _, _ in chosen}) == 208
    (output_dir / 'test_mixed.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r, _, _ in chosen), encoding='utf-8')
    manifest = {'status': 'post-hoc mixture sensitivity; not confirmatory privacy evidence', 'seed': 42,
                'counts': {'synthetic': 100, 'real_authors': 50, 'world_facts': 58},
                'member_predictions': 'new_mia outputs; fixed 200 Forget and 200 Retain',
                'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [new/'test_synthetic.jsonl', old/'test_auxiliary.jsonl']},
                'selected': [{'question': r['question'], 'source': s, 'source_index': i} for r, s, i in chosen]}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    member_groups = {s: load(new / (s + '_eval.jsonl')) for s in ['forget', 'retain']}
    results = []
    for method in ['Vanilla', 'PURE', 'SBU', 'SF-NL', 'FlowMU']:
        a = read_csv(new / (method + '_results.csv'))
        b = read_csv(old / (method + '_results.csv'))
        rows = []
        for split, group in member_groups.items():
            for qa in group:
                row = dict(a[qa['question']])
                assert ' '.join(row['reference_answer'].split()) == ' '.join(qa['answer'].split())
                row.update(split=split, source=split)
                rows.append(row)
        for qa, source, i in chosen:
            origin = a if source == 'synthetic' else b
            row = dict(origin[qa['question']])
            assert ' '.join(row['reference_answer'].split()) == ' '.join(qa['answer'].split())
            row.update(split='test', source=source)
            rows.append(row)
        assert len(rows) == 608
        assert all(r['prediction'].strip() and '[EVAL_ERROR]' not in r['prediction'] for r in rows)
        assert len({r['model_path'] for r in rows if r.get('model_path')}) <= 1
        for row in rows:
            row['rouge_l'] = rouge_l(row['prediction'], row['reference_answer'])
        with (output_dir / (method + '_results.csv')).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader(); writer.writerows(rows)
        for source in ['mixed', 'synthetic', 'auxiliary']:
            selected = [r for r in rows if r['split'] != 'test' or source == 'mixed' or
                        (source == 'synthetic' and r['source'] == 'synthetic') or
                        (source == 'auxiliary' and r['source'] in ['real_authors', 'world_facts'])]
            for pos in ['forget', 'retain']:
                results.append(dict(method=method, nonmember_source=source, **compare(selected, pos, 'test', 'rouge_l')))
    (output_dir / 'mixed_summary.json').write_text(json.dumps({'manifest': manifest, 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output_dir / 'mixed_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    text = ['# 混合非成员 MIA 实测结果', '', '固定 seed=42；100 条 synthetic + 108 条 auxiliary；每个方法共 608 条。复用同一流程的真实预测，没有修改标签、预测或分数。', '',
            '| 方法 | 混合 F-AUC | 混合 R-AUC | Synthetic F-AUC | Auxiliary F-AUC |', '|---|---:|---:|---:|---:|']
    for method in ['Vanilla', 'PURE', 'SBU', 'SF-NL', 'FlowMU']:
        def value(source, pos):
            return next(r['auroc'] for r in results if r['method'] == method and r['nonmember_source'] == source and r['positive_split'] == pos)
        text.append(f"| {method} | {value('mixed','forget'):.4f} | {value('mixed','retain'):.4f} | {value('synthetic','forget'):.4f} | {value('auxiliary','forget'):.4f} |")
    text += ['', '攻击分数为固定 ROUGE-L，较高表示 member-like。该方案在观察历史结果后提出，只能作为探索性的来源混合敏感性分析。',
             '低于 0.5 可能表示分数方向反转；混合后的抵消不能证明成员信号消失。不能将该 ROUGE-L 攻击与参数侧 NLL 学习型攻击混称为同一个设置。',
             '当前 FlowMU 与基线参数训练起点不同，且 synthetic 存在长度混杂，本表不能作为公平参数遗忘的正式结论。置信区间使用逐行 bootstrap，只作探索，未进行作者聚类校正。',
             '成员预测固定复用输入文件；两次生成运行存在输出随机性，因此不把全部历史 AUC 差异都归因于测试来源。']
    (output_dir / '结果.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('\n'.join(text))

if __name__ == '__main__':
    main()
