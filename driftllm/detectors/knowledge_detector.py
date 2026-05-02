from collections import deque
from typing import List, Optional

import torch

from driftllm.detectors.base_detector import BaseDriftDetector, DriftEvent


FIN_PROBES: List[str] = [
    "The Federal Reserve sets the federal funds target range.",
    "A bond price typically falls when yields rise.",
    "EPS means earnings per share.",
    "Treasury bills are short-term government debt instruments.",
    "The S&P 500 is a broad U.S. equity index.",
    "Credit spreads often widen during risk-off periods.",
    "CPI is a common measure of inflation.",
]
ARXIV_PROBES: List[str] = [
    "arXiv papers are organized into subject areas such as cs, math, and stat.",
    "An abstract usually summarizes the problem, method, and results of a paper.",
    "Optimization, representation learning, and systems papers often use different vocabularies.",
    "Machine learning papers on arXiv increased sharply during the mid-2010s.",
    "The primary category of an arXiv paper is often predictive of its language and citations.",
    "Scientific abstracts often contain task-specific terminology and named benchmarks.",
    "Research trends on arXiv shift over time as new methods become popular.",
]

TWEET_PROBES: List[str] = [
    "Social media sentiment changes quickly around major global events.",
    "Hashtags often encode current topics and community identity.",
    "Short posts can carry strong sentiment with little context.",
    "Election periods tend to increase polarization in online language.",
    "Public discourse around AI surged after late 2022.",
    "Breaking news can shift sentiment distributions within days.",
    "Platform-specific slang evolves quickly over time.",
]

AGNEWS_PROBES: List[str] = [
    "Global news coverage intensity can change rapidly during pandemics.",
    "U.S. presidential elections usually drive spikes in world and politics headlines.",
    "Geopolitical conflicts can quickly dominate international news agendas.",
    "Technology coverage increased significantly after major AI product releases in late 2022.",
    "Business news shares can drop when world events dominate front-page attention.",
    "Science and technology sections often absorb AI-related breakthroughs.",
    "Topic distributions in news datasets can shift substantially after global shocks.",
]

AMAZON_PROBES: List[str] = [
    "E-commerce review sentiment can shift during large supply chain disruptions.",
    "Delivery delays often increase negative product reviews.",
    "Consumer product demand changed sharply during COVID lockdown periods.",
    "Post-pandemic shopping patterns affected review language and product focus.",
    "Books and media categories can reflect broader social trend shifts.",
    "New technology product cycles introduce novel terminology in reviews.",
    "Amazon review topic and sentiment distributions vary over calendar years.",
]


class KnowledgeDriftDetector(BaseDriftDetector):
    def __init__(self, domain: str, threshold_pct: float = 0.20):
        probe_map = {
            "financial": FIN_PROBES,
            "tweeteval": TWEET_PROBES,
            "agnews": AGNEWS_PROBES,
            "amazon": AMAZON_PROBES,
            "arxiv": ARXIV_PROBES,
        }
        self.probes = probe_map.get(domain, FIN_PROBES)
        self.threshold_pct = threshold_pct
        self.hist = deque(maxlen=20)

    @torch.no_grad()
    def compute_probe_perplexity(self, model, tokenizer, device) -> float:
        """Return classifier uncertainty on stable probe statements.

        The project uses sequence-classification models for the main task, so
        causal-LM perplexity labels are not available here. Entropy over probe
        logits gives a model-compatible proxy that rises when predictions become
        less confident on domain knowledge probes.
        """
        model.eval()
        entropies = []
        for probe in self.probes:
            enc = tokenizer(probe, return_tensors="pt", truncation=True).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=-1)
            entropies.append(float(entropy.item()))
        return float(torch.tensor(entropies).mean().item())

    def update(self, ppl: float, step: int) -> Optional[DriftEvent]:
        self.hist.append(ppl)
        if len(self.hist) < 10:
            return None
        arr = list(self.hist)
        old = sum(arr[: len(arr) // 2]) / (len(arr) // 2)
        new = sum(arr[len(arr) // 2 :]) / (len(arr) // 2)
        pct = (new - old) / max(1e-8, old)
        if pct > self.threshold_pct:
            return DriftEvent(step, "knowledge_drift", pct, self.threshold_pct, min(1.0, pct), "knowledge_uncertainty")
        return None
