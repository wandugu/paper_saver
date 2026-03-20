from sklearn.metrics import confusion_matrix


def eval_result(true_labels, pred_result, rel2id, logger, use_name=False):
    correct = 0
    total = len(true_labels)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
        if name in rel2id:
            if use_name:
                neg = name
            else:
                neg = rel2id[name]
            break
    for i in range(total):
        if use_name:
            golden = true_labels[i]
        else:
            golden = true_labels[i]

        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0

    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    logger.info('Evaluation result: {}.'.format(result))
    return result


def log_detailed_results(true_labels, pred_labels, rel2id, logger):
    """输出每个类别以及总体的测试情况。"""
    labels = list(rel2id.values())[1:]
    names = list(rel2id.keys())[1:]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    total_samples = int(cm.sum())
    total_success = int(cm.trace())
    total_fail = int(total_samples - total_success)
    success_details = []
    total_details = []
    for idx, name in enumerate(names):
        total = int(cm[idx].sum())
        success = int(cm[idx][idx])
        fail = int(total - success)
        success_details.append(str(success))
        total_details.append(str(total))
        logger.info(f"类别{name}: 总数{total}, 成功{success}, 失败{fail}")
    logger.info(f"总计: 总数{total_samples}, 成功{total_success}, 失败{total_fail}")
    accuracy = float(total_success) / float(total_samples) if total_samples else 0.0
    success_expr = "+".join(success_details) if success_details else "0"
    total_expr = "+".join(total_details) if total_details else "0"
    logger.info(
        "关系判断正确Accuracy = {成功{"
        f"{success_expr}={total_success}" 
        "}}/{总数{"
        f"{total_expr}={total_samples}"
        "}} = "
        f"{accuracy:.4f}"
    )
