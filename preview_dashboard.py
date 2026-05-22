"""Launch dashboard with empty data to preview the UI."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import yaml

with open(ROOT / "config.yml") as f:
    config = yaml.safe_load(f)

from report.dashboard import launch_dashboard

launch_dashboard(
    wfa_results=None,
    mc_results=None,
    sensitivity_results=None,
    regime_results=None,
    stats_results=None,
    config=config,
    verdict_results=None,
    sizing_results=None,
    pre_check_results=None,
)
