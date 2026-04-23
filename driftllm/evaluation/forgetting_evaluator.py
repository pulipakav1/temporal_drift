def forgetting_delta(pre_acc: float, post_acc: float) -> float:
    return max(0.0, pre_acc - post_acc)


def mean_forgetting(deltas):
    if not deltas:
        return 0.0
    return float(sum(deltas) / len(deltas))
