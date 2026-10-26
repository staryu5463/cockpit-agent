"""CockpitAgent: an on-device voice agent for the smart cockpit.

Layers:
  cockpit.env      simulated cockpit (state, catalog, tools, episode)    -> CockpitBench
  cockpit.tasks    task generation, business spec, data splits
  cockpit.verifier private verifier that scores a finished episode
  cockpit.agent    chat format + episode runner shared by every policy
  cockpit.baselines rule-based NLU planner and a fair scripted expert
  cockpit.train    LoRA, SFT, multi-turn GRPO / DAPO
  cockpit.eval     evaluation harness
  cockpit.serve    vLLM client, hybrid router, latency benchmark
  cockpit.rag      owner's-manual retrieval tool
"""

__version__ = "0.1.0"
