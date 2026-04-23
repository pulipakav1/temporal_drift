class BaselineComparison:
    def __init__(self, no_update, full_retrain, selective_ours, oracle=None):
        self.rows = {
            "no_update": no_update,
            "full_retrain": full_retrain,
            "selective_ours": selective_ours,
        }
        if oracle is not None:
            self.rows["oracle"] = oracle

    def print_console_table(self):
        cols = [
            "overall_accuracy",
            "f1_macro",
            "mean_forgetting",
            "n_param_updates",
            "total_adapt_time_seconds",
            "drift_detection_f1",
            "mean_detection_delay_steps",
            "forgetting_bound_violation_rate",
        ]
        print("method | " + " | ".join(cols))
        print("-" * 120)
        for k, v in self.rows.items():
            print(k + " | " + " | ".join([str(v.get(c, "NA")) for c in cols]))

    def print_latex_table(self):
        print("\\begin{tabular}{lcccccccc}")
        print("\\toprule")
        print("Method & Acc & F1 & Forget & ParamUpd & Time(s) & DriftF1 & Delay & ViolRate \\\\")
        print("\\midrule")
        for k, v in self.rows.items():
            print(
                f"{k} & {v.get('overall_accuracy','NA')} & {v.get('f1_macro','NA')} & "
                f"{v.get('mean_forgetting','NA')} & {v.get('n_param_updates','NA')} & "
                f"{v.get('total_adapt_time_seconds','NA')} & {v.get('drift_detection_f1','NA')} & "
                f"{v.get('mean_detection_delay_steps','NA')} & {v.get('forgetting_bound_violation_rate','NA')} \\\\"
            )
        print("\\bottomrule")
        print("\\end{tabular}")


def from_baseline_aggregate(aggregate_dict):
    return BaselineComparison(
        aggregate_dict.get("no_update", {}),
        aggregate_dict.get("full_retrain", {}),
        aggregate_dict.get("selective_ours", {}),
        aggregate_dict.get("oracle", None),
    )
