"""Write source-linked Chinese conclusions, embed generated tables, and seal evidence."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re

from .phase3_replay import json_read
from .protocol import digest, read_records


_PYTEST_COUNT = r'\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)'
_PYTEST_SUMMARY = re.compile(rf'{_PYTEST_COUNT}(?:, {_PYTEST_COUNT})*(?: in .+)?')
_R4_SETTINGS = tuple(f'h{family}_c{fraction:02d}' for family in (1, 3, 5) for fraction in (2, 5, 10, 20))


def validation_summaries(log):
    """Read completed pytest summaries without treating collection as execution."""
    results = []
    for line in log.splitlines():
        line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip(' =\t')
        if not _PYTEST_SUMMARY.fullmatch(line):
            continue
        counts = {name: int(count) for count, name in re.findall(r'(\d+) ([a-z]+)', line.split(' in ', 1)[0])}
        if any(name in counts for name in ('passed', 'failed', 'error', 'errors', 'skipped', 'xfailed', 'xpassed')):
            results.append({'passed': counts.get('passed', 0), 'failed': counts.get('failed', 0),
                            'errors': counts.get('error', 0) + counts.get('errors', 0)})
    return results


def replay_findings(metrics):
    savings = []
    for seed, metric in sorted(metrics.items()):
        result = metric['r1']['d3']['test']
        saved = round(result['n_posts'] * (1 - result['detail']))
        savings.append(f"seed{seed}：相对每帖一次详细编码减少 {saved} 次（{100 * (1 - result['detail']):.5f}%），F1 regret={result['f1_regret']:.5f}")
    fallbacks = {
        setting: [seed for seed, metric in sorted(metrics.items()) if metric['r4'][setting]['choices']['d3']['fallback']]
        for setting in _R4_SETTINGS
    }
    return '；'.join(savings), '；'.join(
        f"{setting}：{len(seeds)}/{len(metrics)}，种子 {', '.join(map(str, seeds)) or '无'}"
        for setting, seeds in fallbacks.items()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--validation-log', required=True, type=Path)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    rows = read_records(evidence / 'ledger.jsonl')
    audits = {row['seed']: row for row in rows if row['experiment'] == 'phase3_audit'}
    if set(audits) != {42, 43, 44} or any(row['status'] != 'SUCCESS' for row in audits.values()):
        raise ValueError('Three registered successful seeds required')
    metrics = {seed: json_read(evidence / row['run_dir'] / 'metrics.json') for seed, row in audits.items()}
    verifies = [row for row in rows if row['experiment'] == 'phase3_verify' and row['status'] == 'SUCCESS']
    if len(verifies) != 1:
        raise ValueError('One R5 source audit required')
    verify = verifies[0]
    verified = json_read(evidence / verify['run_dir'] / 'metrics.json')
    router_audit = json_read(evidence / verify['run_dir'] / 'learned_router_code_audit.json')
    for seed, metric in metrics.items():
        if set(metric['r4']) != set(_R4_SETTINGS) or len(metric['r1']) != 16:
            raise ValueError('Missing preregistered selector/sensitivity configuration')
        provenance = json_read(evidence / audits[seed]['run_dir'] / 'test_provenance.json')
        if provenance['new_test_forward_passes'] != 0:
            raise ValueError('Unexpected test rerun')
    now = datetime.now(timezone(timedelta(hours=8))).isoformat()
    def table(name):
        return (evidence / 'tables' / (name + '.md')).read_text(encoding='utf-8')
    savings, fallbacks = replay_findings(metrics)
    log_text = args.validation_log.read_text(encoding='utf-8')
    validation_counts = validation_summaries(log_text)
    validation_result = '；'.join(f"通过 {row['passed']}，失败 {row['failed']}，错误 {row['errors']}"
                                 for row in validation_counts)
    validation_result = (f'验证日志中的 pytest 汇总：{validation_result}。' if validation_result else
                         '验证日志没有 pytest 执行汇总，无法确认测试结果。')
    baseline_differences = '；'.join(
        f"seed{seed} ΔF1={metric['r2']['test']['all_off_minus_independent']['f1_delta']:.5f}，帖子分歧={metric['r2']['test']['all_off_minus_independent']['post_disagreement']:.5f}"
        for seed, metric in sorted(metrics.items()))
    deployment_differences = '；'.join(
        f"seed{seed} 相对独立文本 ΔF1={metric['r2']['test']['deploy']['d3']['versus_independent']['f1_delta']:.5f}"
        for seed, metric in sorted(metrics.items()))
    phase_counts = Counter(row['experiment'] for row in rows)
    phase3 = [row for row in rows if row['experiment'].startswith('phase3_')]
    phase3_metrics = []
    for row in phase3:
        path = evidence / row['run_dir'] / 'metrics.json'
        phase3_metrics.append(json_read(path) if path.is_file() else {})
    resources = {name: sum(item[name] for item in phase3_metrics) if all(name in item for item in phase3_metrics) else None
                 for name in ('gpu_hours_actual', 'training_updates')}
    timing = verified['timing']
    router_configs = router_audit['routers']
    same_gate = router_configs['utility_mlp']['architecture'] == router_configs['saver']['architecture']
    grouping = '；'.join(f"{seed}：fit/calibration 重叠帖 {item['fit_calibration_overlap']}，"
                        f"fit/dev 截断帖 {item['fit_partial_posts']}/{item['dev_partial_posts']}"
                        for seed, item in sorted(verified['grouping'].items()))
    summary = [
        '# P24 第三阶段真实运行摘要', '', f'生成时间：{now}。所有数值来自 ledger/metrics 与 make_tables 输出。', '',
        f'- R1：按开发集效用选择的主策略在测试集上的详细编码差额：{savings}。差额不代表实测延迟。[p24_phase3_R1_d3.md；p24_phase3_R1_l5.md]',
        '- R1：各选择规则及其 all-off 和 fallback 结果见生成表。[p24_phase3_R1_max_coverage.md；p24_phase3_R1_max_dev_f1.md；p24_phase3_R1_d5.md]',
        f'- R2：same-checkpoint all-off 与独立 ModernBERT-only 的配对差异：{baseline_differences}。[p24_phase3_R2_test.md]',
        f'- R2：{deployment_differences}。保证仅相对 same-checkpoint all-off。[p24_phase3_R2_test_d3.md]',
        '- R3：帖子 AUROC、排序一致率和单位 CE 代理结果分别列示。[p24_phase3_R3_dev.md；p24_phase3_R3_test.md]',
        f'- R4：主策略的回退频率和回退种子：{fallbacks}。',
        f"- R5：tokens+image ID 的 distinct posts 为 train={verified['dataset']['canonical_post']['train']['posts']}、dev={verified['dataset']['canonical_post']['dev']['posts']}、test={verified['dataset']['canonical_post']['test']['posts']}，跨官方分区={verified['dataset']['canonical_post']['cross_split_posts']}；仅文本相同则有 {verified['dataset']['text']['cross_split_posts']} 组跨分区。[p24_phase3_R5_posts_canonical_post.md；p24_phase3_R5_posts_text.md]",
        f'- R5：实际分组核验：{grouping}。[p24_phase3_R5_grouping_s42.md；p24_phase3_R5_grouping_s43.md；p24_phase3_R5_grouping_s44.md]',
        f"- R5：审计记录中 UtilityMLP 与 SAVER 的 gate 结构相同：{'是' if same_gate else '否'}。[{verify['run_id']}/learned_router_code_audit.json]",
        f"- R5：计时日志 {timing['logged_runs']} 个，含组件计时的日志 {timing['component_timed_runs']} 个。R1 延迟来自端点重放。[p24_phase3_R5_latency_components.md；p24_phase3_R1_l5.md]",
        f"- 第三阶段台账条目 {len(phase3)} 个，记录的 GPU 小时：{resources['gpu_hours_actual']}，训练更新：{resources['training_updates']}。{validation_result}",
    ]
    (evidence / 'real_runs_summary.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    validation = evidence / 'provenance' / 'validation.log'
    validation.write_text(log_text, encoding='utf-8')
    report = [
        '# P24 SAVER 第三阶段：冻结 checkpoint 重放与评测报告', '', f'完成时间：{now}（北京时间）。', '',
        '报告汇总 R1–R4 重放与 R5 数据和代码核验。下列表格嵌入 `python -m repro.make_tables` 的 Markdown 产物，F1、harm 和成本比例均为 0–1 口径。', '',
        '## 1. 预注册与运行范围', '',
        '- 预注册固定 seed42/43/44、101 阈值、harm 上界 α=0.10、δ=0.05、Bonferroni CP，以及 λ 和敏感性全部网格。保证逐 seed 条件成立；三种子均值不继承联合置信保证。',
        '- distinct λ 网格为 [0, .01, .05, .10, .25, .50, 1]，主项 .10；latency λ/ms 网格为 [0, .0001, .0005, .001, .005, .01, .05]，主项 .01。dev 效用为相对 same-checkpoint all-off 的 micro-F1 增益减去 λ×成本。成本分别为帖子内去重详细图片数和测量端点重放延迟。',
        '- dev 使用全部官方关系对并按整帖统计；calibration 复用原 holdout。20% 扩展只从该 checkpoint 未参与实际 fit 的训练帖子补入，完整保留其所有关系对；不是更换训练划分或重新训练。',
        '- 所有实验通过 `python -m repro.run` 启动；R1/R4 的全部配置锁写入 METHOD_LOCK 后才载入 test 标签。复用 phase2 forced-branch test 与独立文本 test，无新 test forward；相同动作掩码只评分一次。历史 test 已被先前阶段观察，不能称为全新未见 test。',
        f'- CPU 补算时间与 T4 延迟分别报告。{validation_result}原日志见 `provenance/validation.log`。', '',
        'phase3 run_id：', '',
    ]
    report.extend(f"- `{row['run_id']}`：{row['experiment']}，seed={row['seed']}，{row['status']}，code={row['git_commit']}。" for row in phase3)
    report.extend(['', '## 2. R1 风险—价值—成本选择', '',
        '下表对照现规则、max dev F1 和主 value_distinct。每帖不同详细图片数是真正的附件并集；不是直接把激活关系对比例当成本。CalUCB 对正帖子 harm 的 Bernoulli 包络应用精确 CP，因此上界也覆盖归一化平均正 harm。all-off 的 harm 恒为零，无需估计。', '',
        table('p24_phase3_R1_max_coverage'), '', table('p24_phase3_R1_max_dev_f1'), '',
        table('p24_phase3_R1_d3'), '', table('p24_phase3_R1_l5'), '',
        f'主策略测试集详细编码差额：{savings}。', '',
        'F1 regret 是相对同一 split 的最优可行 F1；latency regret 是相对最小可行延迟，通常由 all-off 取得。二者是各自单目标的描述性 oracle，不能解释为同一个可部署最优点。metrics 还保留 F1 不低于所选策略条件下的 latency regret。test oracle 与 Pareto 仅作锁定后的描述，绝不反选阈值。', '',
        '全部可行阈值的 dev/test F1–Harm–D/post–latency 数据及 Pareto 标志保存在每个 phase3_audit 的 `feasible_frontier.json`，并逐项由 `p24_phase3_frontier_s{seed}_{dev|test}_*.md/.tex` 导出。全部 λ 见 R1_d0–d6、R1_l0–l6 表。', '',
        'latency 的限制：使用 phase2 GPU0 历史批量服务日志中的相同整帖、batch=1、未分桶、三次 pass 端点；本阶段只读日志。off 是仍做 64-patch 全图筛查的 SAVER all-off 实现，on 是匹配架构的独立 always-vision checkpoint；二者未逐组件计时。按测得端点均值对策略成本重放，未测的策略延迟和 P90 只能标为 proxy，不能写成新策略直接实测加速。', '',
        table('p24_phase3_R1_d5'), '',
        '## 3. R2 文本参考路径审计', '',
        '独立 ModernBERT-only checkpoint 三个种子均存在；沿用第一阶段冻结 backbone、独立训练 task head 的基线定义。dev 为本轮 CPU 补推理，test 原日志复用。paired CI 用整帖有放回 bootstrap，2000 draws；不是对三种子联合显著性的声明。', '',
        table('p24_phase3_R2_dev'), '', table('p24_phase3_R2_test'), '',
        table('p24_phase3_R2_dev_d3'), '', table('p24_phase3_R2_test_d3'), '',
        f'{baseline_differences}。{deployment_differences}。证书的参照是同一检查点的 all-off 路径。', '',
        '逐帖预测、gold、门控分数与分歧可在 `post_predictions_dev.json` / `post_predictions_test.json` 查到；帖子为不含原文本的稳定 hash，全部同帖关系对仍成组。完整配对增益/损害、参考差异和每个 λ 的部署差异在 metrics.r2，主对照另有生成表。', '',
        '## 4. R3 单元排序与帖子 gain', '',
        '使用 forced-on 对 all-off 的最终帖子归一化准确率增益；排序分数为帖子内 unit gate 均值。AUROC 将 gain>0 视作正类，gain=0 为负；RankAgree 对 gain 不同的帖子对统计，分数并列给半分。UnitAUROC 按单元分数预测其 parent-post gain。', '',
        table('p24_phase3_R3_dev'), '', table('p24_phase3_R3_test'), '',
        'SignAgree 受帮助帖占比和 gate 饱和影响。结果范围为 MNRE。', '',
        '## 5. R4 策略族与校准比例敏感性', '',
        'harm penalty 集为 [0]、[0,.5,1]、[0,.25,.5,1,2]，分数明确使用 g−η(1−g)。没有可复用的独立 harm head，因此这是预声明代理 penalty，且只重参数化同一排序；不能包装成 learned harm-aware 策略族。Nominal 为 101×penalty 个数，置信修正按名义大小；Effective 为校准人口上真实不同的动作掩码数量（含 exact off）。', '',
        f'主策略的回退频率和回退种子：{fallbacks}。20% 从未 fit 的训练帖子扩展，是规模敏感性分析。', '',
    ])
    for count in (1, 3, 5):
        for fraction in (2, 5, 10, 20):
            report.extend([table(f'p24_phase3_R4_h{count}_c{fraction:02d}'), ''])
    report.extend(['三种子 fallback 均值的生成表以 `_aggregate.md` 命名。UCBSlack 是上界减去经验正 harm 帖概率。', '',
        '## 6. R5 数据、分组、计时与 learned router 核实', '',
        table('p24_phase3_R5_posts_canonical_post'), '', table('p24_phase3_R5_posts_text'), '',
        f'实际分组核验：{grouping}。官方分区交叉情况见生成表。', '',
    ])
    for seed in (42, 43, 44):
        report.extend([table(f'p24_phase3_R5_grouping_s{seed}'), ''])
    report.extend(['旧代码先按帖划 fit/calibration，再对 fit 截 4096 pair、dev 截 512 pair，因而 fit/dev 内有部分帖只保留部分关系对。这没有造成同帖跨 fit/calibration，却意味着“所有旧 fit/dev 都是完整帖子”不成立。本轮已补齐整个官方 dev；不改变冻结 checkpoint 的训练历史。', '',
        table('p24_phase3_R5_latency_components'), '',
        f"计时日志 {timing['logged_runs']} 个，含组件计时的日志 {timing['component_timed_runs']} 个。原始列名和日志 SHA256 见 `latency_log_audit.json`。", '',
        '实际代码中的 learned routers：', '',
    ])
    for router in ('linear', 'utility_mlp', 'saver'):
        report.extend([table(f'p24_phase3_R5_router_{router}'), ''])
    for name, config in sorted(router_configs.items()):
        dimensions = verified['routers'][name]['input_features']
        teachers = ', '.join(config['teacher_variants']) or '无记录'
        report.append(f"- {name}：输入维度 {dimensions}，gate 参数 {config['gate_parameters']}，教师 {teachers}。"
                      f"损失：`{config['loss']}`。")
    report.extend(['', f"代码出处绑定 commit `{router_audit['source_commit']}` 的文件 SHA256：", ''])
    for filename, item in router_audit['files'].items():
        report.append(f"- `repro/p24/{filename}`，SHA256 `{item['sha256']}`：")
        for source_line in item['evidence']:
            report.append(f"  - `{source_line['file']}:{source_line['line']}`：`{source_line['code']}`")
    report.extend(['', '## 7. 证据包、复核与资源', '',
        f'证据目录：`{evidence}`。', '',
        f'- ledger 合并 {len(rows)} 个唯一 run，含阶段1/2、阶段3和最初已废弃 pilot；保留原 host、状态、来源 ledger 与 git commit。各实验计数：`{dict(sorted(phase_counts.items()))}`。历史 BLOCKED 与 SUPERSEDED 标记保留，不纳入成功结果。',
        '- `runs/<run_id>/`：metrics、config、env、gates、缺失/废弃标记、输入/方法锁、逐帖预测或去概率张量的 compact replay JSON、原始全部产物路径/大小/SHA256。没有复制 checkpoint、训练轨迹或原图；原始文件保存在运行主机，原始 hash 不冒充 compact 文件 hash。',
        '- `tables/`：只由 make_tables 生成，含完成后重建的阶段1/2结果、全部阶段3 λ、各 seed UCB、敏感性和可行 frontier 的 Markdown/LaTeX。',
        '- `provenance/`：三版预注册、重放/表格/导出代码、commit、来源台账收据与测试日志；`real_runs_summary.md` 中文结论逐条链接 run/table；`MANIFEST.json` 对包内全部其他文件计算 SHA256。',
        '## 8. 结果局限', '',
        '- 帖级详细编码差额和端点延迟重放均不代表新策略的实测加速。',
        '- 风险界以同一检查点的 all-off 为参照；相对于独立纯文本模型的比较单独报告。',
        '- harm penalty 仅使用代理分数。已有日志只支持端到端计时，缺少组件计时。',
        '- 多图与 MNER 实验缺失。第一阶段的 hard/last 检查点比较仍为 BLOCKED；第二阶段的前瞻 E6 单独报告。', '',
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(report), encoding='utf-8')
    files = [{'path': str(path.relative_to(evidence)), 'bytes': path.stat().st_size, 'sha256': digest(path)}
             for path in sorted(evidence.rglob('*')) if path.is_file() and path.name != 'MANIFEST.json']
    manifest = {'created_at': now, 'files': files, 'manifest_self_excluded': True, 'ledger_runs': len(rows),
                'total_bytes_excluding_manifest': sum(item['bytes'] for item in files),
                'validation': {'all_seed_runs': True, 'all_R4_settings': True, 'all_R1_selectors': True,
                               'new_test_forward_passes': 0, 'phase3_gpu_hours': resources['gpu_hours_actual'],
                               'phase3_training_updates': resources['training_updates'], 'pytest_summaries': validation_counts,
                               'numeric_tables_generator': 'python -m repro.make_tables'}}
    (evidence / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    total = sum(path.stat().st_size for path in evidence.rglob('*') if path.is_file())
    largest = max(path.stat().st_size for path in evidence.rglob('*.json'))
    if total > 30000000 or largest >= 3000000:
        raise ValueError(f'Export limits exceeded: total={total}, largest JSON={largest}')
    for item in manifest['files']:
        if digest(evidence / item['path']) != item['sha256']:
            raise ValueError(f'Hash verification failed: {item["path"]}')
    with args.report.open('a', encoding='utf-8') as stream:
        stream.write(f'\n交付校验：证据包 {total:,} 字节，{len(files) + 1} 个文件，最大 JSON {largest:,} 字节，'
                     f'{len(rows)} 个唯一 run；逐文件 SHA256 复核 0 ERROR。MANIFEST 自身不纳入其文件列表。\n')
    print(json.dumps({'report': str(args.report), 'evidence': str(evidence), 'bytes': total,
                      'files': len(files) + 1, 'largest_json_bytes': largest, 'runs': len(rows),
                      'report_sha256': digest(args.report), 'hash_errors': 0}))


if __name__ == '__main__':
    main()
