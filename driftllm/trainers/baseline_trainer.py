from driftllm.trainers.online_trainer import OnlineDriftTrainer


class NoUpdateBaseline:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        return OnlineDriftTrainer(self.cfg, mode="no_update").run()


class FullRetrainBaseline:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        return OnlineDriftTrainer(self.cfg, mode="full_retrain").run()


class OracleBaseline:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self):
        return OnlineDriftTrainer(self.cfg, mode="oracle").run()


def run_baselines(cfg):
    no_update = NoUpdateBaseline(cfg).run()
    full = FullRetrainBaseline(cfg).run()
    selective = OnlineDriftTrainer(cfg, mode="selective_ours").run()
    oracle = OracleBaseline(cfg).run()
    return {"no_update": no_update, "full_retrain": full, "selective_ours": selective, "oracle": oracle}
