import argparse
import ast
import csv
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path


getcontext().prec = 50
HERE = Path(__file__).resolve().parent
RATIOS = ('1%', '5%', '10%')
METHODS = ('Vanilla', 'PURE', 'SBU', 'SF-NL', 'FlowMU (GA)',
           'FlowMU (DPO)', 'FlowMU (RMU)')
OLD_PLOT = HERE.parent / 'figure/uts_performance_allmethods_legend_1p5x/plot_uts_performance_allmethods.py'


def load_table(path):
    data = {}
    dataset = None
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        cells = [cell.strip() for cell in line.split('\t')]
        if cells[0] in ('TOFU', 'LiteraryQA'):
            dataset = cells[0]
            if dataset in data:
                raise ValueError(f'Duplicate dataset at line {line_number}')
            data[dataset] = {}
        elif cells[0] in METHODS:
            if dataset is None or len(cells) != 13:
                raise ValueError(f'Invalid data row at line {line_number}')
            if cells[0] in data[dataset]:
                raise ValueError(f'Duplicate method at line {line_number}')
            values = tuple(Decimal(cell) for cell in cells[1:])
            if not all(value.is_finite() and 0 <= value <= 100 for value in values):
                raise ValueError(f'Invalid percentage at line {line_number}')
            data[dataset][cells[0]] = dict(zip(RATIOS, (values[i:i+4] for i in (0, 4, 8))))
        elif any(cells) and not cells[0].startswith(('orget Ratio', 'Forget Ratio', 'Task')):
            raise ValueError(f'Unknown row at line {line_number}: {line}')
    if set(data) != {'TOFU', 'LiteraryQA'}:
        raise ValueError('Expected TOFU and LiteraryQA')
    for rows in data.values():
        if set(rows) != set(METHODS):
            raise ValueError('Missing or unexpected methods')
    return data


def verify_old_plot(data, path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    raw = next(ast.literal_eval(node.value) for node in tree.body
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'RAW' for target in node.targets))
    for dataset, rows in data.items():
        for method, ratios in rows.items():
            for ratio, values in ratios.items():
                expected = tuple(Decimal(str(v)) for v in raw[dataset][method][ratio])
                if values != expected:
                    raise ValueError(f'Old plot mismatch: {dataset}/{method}/{ratio}')


def uts_new(forget, retain, baseline_retain):
    if baseline_retain <= 0:
        raise ValueError('Vanilla retain baseline must be positive')
    a = Decimal(100) - forget
    b = Decimal(100) * retain / baseline_retain
    if a + b == 0:
        raise ValueError('UTS is undefined when both terms are zero')
    result = 2 * a * b / (a + b)
    alternate = 2 * (100-forget) * retain * 100 / ((100-forget)*baseline_retain + retain*100)
    if abs(result - alternate) > Decimal('1e-40'):
        raise ArithmeticError('Independent formula check failed')
    return result


def rounded(value):
    return str(value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))


def main():
    parser = argparse.ArgumentParser(description='Compute unclipped UTS from the percentage table.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--old-plot', type=Path, default=OLD_PLOT)
    args = parser.parse_args()
    data = load_table(args.input)
    verify_old_plot(data, args.old_plot)
    assert uts_new(Decimal(0), Decimal(50), Decimal(50)) == 100
    assert uts_new(Decimal(100), Decimal(50), Decimal(50)) == 0
    rows = []
    for dataset, methods in data.items():
        for method in METHODS:
            for ratio in RATIOS:
                f, r, fr, rr = methods[method][ratio]
                _, r0, _, rr0 = methods['Vanilla'][ratio]
                acc = uts_new(f, r, r0)
                rouge = uts_new(fr, rr, rr0)
                rows.append(dict(Dataset=dataset, Method=method, Forget_Ratio=ratio,
                                 F_ACC=str(f), R_ACC=str(r), F_RougeL=str(fr), R_RougeL=str(rr),
                                 R0_ACC=str(r0), R0_RougeL=str(rr0),
                                 UTS_ACC=rounded(acc), UTS_RougeL=rounded(rouge),
                                 UTS_ACC_full=str(acc), UTS_RougeL_full=str(rouge)))
    assert len(rows) == 42
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'uts_new_results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(formula='2*(100-F)*(100*R/R0)/((100-F)+(100*R/R0))',
                  baseline='Vanilla retain score of the same dataset, ratio and metric',
                  units='percent (0 to 100) for input scores',
                  rounding='Decimal precision=50; ROUND_HALF_UP to 2 decimals only at output',
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  verification='168 input values match old plotting RAW; image manually checked against input',
                  old_plot=str(args.old_plot), results=rows)
    (args.output_dir / 'uts_new_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# 无截断 UTS 计算结果', '',
             '`UTS_new = 2*(100-F)*(100*R/R0)/((100-F)+(100*R/R0))`', '',
             'R0 为同数据集、同切分的 Vanilla R-ACC 或 R-RougeL。输入为百分数，不截断保留项。',
             '图片与输入表的 168 个值经人工核对一致，脚本另与旧绘图 RAW 数据逐项核对。',
             '使用表中已保留两位小数的输入计算，并非从未舍入的逐样本结果计算。', '']
    for dataset in data:
        lines += [f'## {dataset}', '',
                  '| 方法 | 1% ACC-UTS | 1% RougeL-UTS | 5% ACC-UTS | 5% RougeL-UTS | 10% ACC-UTS | 10% RougeL-UTS |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for method in METHODS:
            selected = [row for row in rows if row['Dataset'] == dataset and row['Method'] == method]
            values = [row[key] for row in selected for key in ('UTS_ACC', 'UTS_RougeL')]
            lines.append('| ' + ' | '.join([method] + values) + ' |')
        lines.append('')
    lines += ['## 旧绘图代码', '', f'`{args.old_plot}`', '',
              '旧函数 `uts()` 使用 min(R/R0, 1)，此次未修改旧绘图脚本或原始数据。']
    content = '\n'.join(lines)+'\n'
    (args.output_dir / 'uts_new_results.md').write_text(content, encoding='utf-8')
    print(content)


if __name__ == '__main__':
    main()
