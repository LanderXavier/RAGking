"""
ui/helpers.py — Terminal colors, prompts, banners, and shared UI utilities.
"""

import os
from core.config import state, tr


# ══════════════════════════════════════════════════════════════════════════════
#  ANSI COLORS
# ══════════════════════════════════════════════════════════════════════════════
class C:
    R  = "\033[0m"
    B  = "\033[1m"
    CY = "\033[96m"
    GR = "\033[92m"
    YL = "\033[93m"
    RD = "\033[91m"
    GY = "\033[90m"
    WH = "\033[97m"
    BL = "\033[94m"
    MG = "\033[95m"


def clr():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    clr()
    print(f"""{C.CY}{C.B}
╔══════════════════════════════════════════════════════════════════════════════╗
║  ██████╗  █████╗  ██████╗ ██╗  ██╗██╗███╗  ██╗ ██████╗                   ║
║  ██╔══██╗██╔══██╗██╔════╝ ██║ ██╔╝██║████╗ ██║██╔════╝                   ║
║  ██████╔╝███████║██║  ███╗█████╔╝ ██║██╔██╗██║██║  ███╗                  ║
║  ██╔══██╗██╔══██║██║   ██║██╔═██╗ ██║██║╚████║██║   ██║                  ║
║  ██║  ██║██║  ██║╚██████╔╝██║  ██╗██║██║ ╚███║╚██████╔╝                  ║
║  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚══╝ ╚═════╝                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  RAG Evaluation & Ranking Framework  ·  v2.1.0  ·  by Lander Liquicota     ║
╚══════════════════════════════════════════════════════════════════════════════╝{C.R}""")


def section(title: str):
    pad = 70 - len(title)
    print(f"\n{C.BL}{'─'*5} {C.B}{C.WH}{title}{C.R}{C.BL} {'─'*max(pad,0)}{C.R}")


def ok(msg):    print(f"  {C.GR}{C.B}✔{C.R}  {msg}")
def err(msg):   print(f"  {C.RD}{C.B}✘{C.R}  {C.RD}{msg}{C.R}")
def warn(msg):  print(f"  {C.YL}{C.B}⚠{C.R}  {C.YL}{msg}{C.R}")
def info(msg):  print(f"  {C.CY}ℹ{C.R}  {msg}")


def ask(prompt: str, default="") -> str:
    dflt = f" {C.GY}[{default}]{C.R}" if default != "" else ""
    val = input(f"\n  {C.YL}▶{C.R} {prompt}{dflt}: ").strip()
    return val if val else str(default)


def ask_float(prompt: str, default) -> float:
    while True:
        try:
            return float(ask(prompt, default))
        except ValueError:
            err("Enter a valid number.")


def ask_float_range(prompt: str, default, min_val=None, max_val=None) -> float:
    while True:
        val = ask_float(prompt, default)
        if min_val is not None and val < min_val:
            err(f"Value must be >= {min_val}.")
            continue
        if max_val is not None and val > max_val:
            err(f"Value must be <= {max_val}.")
            continue
        return val


def ask_int_range(prompt: str, default, min_val=None, max_val=None) -> int:
    while True:
        raw = ask(prompt, default)
        try:
            val = int(raw)
        except ValueError:
            err("Enter a whole number.")
            continue
        if min_val is not None and val < min_val:
            err(f"Value must be >= {min_val}.")
            continue
        if max_val is not None and val > max_val:
            err(f"Value must be <= {max_val}.")
            continue
        return val


def pause():
    input(f"\n  {C.GY}Press Enter to continue...{C.R}")


def status_badge():
    loaded = (
        f"{C.GR}● {tr('status_loaded')}  ({len(state['df'])} rows){C.R}"
        if state["df"] is not None
        else f"{C.RD}○ {tr('status_not_loaded')}{C.R}"
    )
    scored = (
        f"{C.GR}● {tr('status_results_ready')}{C.R}"
        if state["results_df"] is not None
        else f"{C.GY}○ {tr('status_not_evaluated')}{C.R}"
    )
    fw_count = len(state.get("frameworks", {}))
    fw_badge = f"  {C.MG}● {fw_count} framework(s) loaded{C.R}" if fw_count else ""
    print(f"  {C.GY}Status:{C.R}  {loaded}   {scored}{fw_badge}\n")


def _warn_once(tag, message: str):
    warned = state.setdefault("_warned", set())
    if tag in warned:
        return
    warn(message)
    warned.add(tag)


def _looks_like_placeholder_secret(value) -> bool:
    if value is None:
        return True
    v = str(value).strip().lower()
    if not v:
        return True
    return (
        "your_" in v
        or v.endswith("_here")
        or "replace" in v
        or "changeme" in v
        or "example" in v
    )


def check_api_key(provider: str) -> bool:
    from core.config import KNOWN_PROVIDERS
    if provider == "local":
        return True
    key_env = KNOWN_PROVIDERS.get(provider, {}).get("key_env")
    if not key_env:
        return False
    key = os.getenv(key_env, "")
    if not (key and len(key) > 10):
        return False
    return not _looks_like_placeholder_secret(key)
