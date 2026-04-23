def recovery_steps(acc_series, target: float):
    for i, a in enumerate(acc_series):
        if a >= target:
            return i
    return -1


def recovery_steps_to_fraction(post_event_acc_series, pre_event_acc: float, fraction: float = 0.9):
    target = pre_event_acc * fraction
    return recovery_steps(post_event_acc_series, target=target)
