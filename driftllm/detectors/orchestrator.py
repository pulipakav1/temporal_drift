from typing import Optional

from driftllm.detectors.base_detector import DriftEvent


class DriftOrchestrator:
    def __init__(self, semantic, label, knowledge, cooldown_steps: int = 300):
        self.semantic = semantic
        self.label = label
        self.knowledge = knowledge
        self.cooldown_steps = cooldown_steps
        self.last_fire_step = -10**9

    def update(self, step: int, embedding=None, pred_label=None, ppl=None) -> Optional[DriftEvent]:
        if step - self.last_fire_step < self.cooldown_steps:
            return None
        events = []
        if embedding is not None:
            e = self.semantic.update(embedding, step)
            if e:
                events.append(e)
        if pred_label is not None:
            e = self.label.update(pred_label, step)
            if e:
                events.append(e)
        if ppl is not None:
            e = self.knowledge.update(ppl, step)
            if e:
                events.append(e)
        if not events:
            return None
        top = sorted(events, key=lambda x: x.severity, reverse=True)[0]
        self.last_fire_step = step
        return top
