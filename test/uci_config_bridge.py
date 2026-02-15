"""
uci_config_bridge.py

Auto-registers config.py parameters as UCI options so that cutechess-cli
can configure each engine instance via  option.PARAM=value .

Integration (add to uci.py after defining UCI_TUNABLES):
    from uci_config_bridge import register_config_tunables
    register_config_tunables(UCI_TUNABLES, UCIOption)
"""
from typing import Any, Callable, Dict

import config
from chess_engine import configure_nn_type
#from config import configure_multi_core_blas


class UCIOption:
    def __init__(
        self,
        name: str,
        opt_type: str,
        default: Any,
        min_val: Any = None,
        max_val: Any = None,
        apply: Callable[[Any], None] = None,
    ):
        self.name = name
        self.type = opt_type  # "spin", "check", "string"
        self.default = default
        self.min = min_val
        self.max = max_val
        self.apply = apply

    def uci_declaration(self) -> str:
        if self.type == "spin":
            return (
                f"option name {self.name} type spin "
                f"default {self.default} min {self.min} max {self.max}"
            )
        elif self.type == "check":
            default = "true" if self.default else "false"
            return f"option name {self.name} type check default {default}"
        elif self.type == "string":
            return f"option name {self.name} type string default {self.default}"
        else:
            raise ValueError(f"Unknown UCI option type: {self.type}")

    def parse_value(self, value: str):
        if self.type == "spin":
            return int(value)
        elif self.type == "check":
            return value.lower() == "true"
        elif self.type == "string":
            return value
        else:
            return value


UCI_TUNABLES: Dict[str, UCIOption] = {}

# ---------- classification ----------

_FLOAT_PARAMS = {
    'QS_SOFT_STOP_DIVISOR', 'QS_TIME_CRITICAL_FACTOR',
    'EMERGENCY_TIME_RESERVE', 'ESTIMATED_BRANCHING_FACTOR',
    'TIME_SAFETY_MARGIN_RATIO', 'QS_TIME_BUDGET_FRACTION',
}

_LIST_PARAMS = {
    'MAX_QS_MOVES', 'MAX_QS_MOVES_DIVISOR',
    'FUTILITY_MARGIN', 'RAZORING_MARGIN',
}

# Not tunable or already handled by dedicated UCI options
_SKIP = {
    'MAX_SCORE', 'TANH_SCALE',              # fundamental constants
    'DIAGNOSTIC', 'debug_mode',           # debug flags
    'QUIET_CONFIG',
    'PONDERING_ENABLED',                  # UCI Ponder
    'MAX_THREADS',                           # UCI Threads
    "RESIGN_THRESHOLD",
    "RESIGN_CONSECUTIVE_MOVES",
    #'MULTI_CORE_BLAS',                       # already registered manually
}

# ---------- helpers ----------

def _make_apply(key, cast_fn):
    """Return an apply callback that sets config.<key> = cast(value)."""
    def _apply(value):
        setattr(config, key, cast_fn(value))
        print(f"info string {key}={getattr(config, key)}", flush=True)
    return _apply


def _smart_parse(key, value_str):
    """Parse string → list / float / str depending on param category."""
    if key in _LIST_PARAMS:
        # e.g. "[12,6,4,2]"
        return eval(value_str)
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return value_str


def _register_spin(d, cls, key, default):
    # max must be >= default; use 10x default or 100k, whichever is larger
    max_val = max(100_000, abs(default) * 10)
    d[key] = cls(
        name=key, opt_type="spin", default=default,
        min_val=0, max_val=max_val,
        apply=_make_apply(key, int),
    )


def _register_check(d, cls, key, default):
    d[key] = cls(
        name=key, opt_type="check", default=default,
        apply=_make_apply(key, lambda v: v if isinstance(v, bool)
                          else str(v).lower() in ('true', '1')),
    )


def _register_string(d, cls, key, default_str):
    d[key] = cls(
        name=key, opt_type="string", default=default_str,
        apply=_make_apply(key, lambda v: _smart_parse(key, v)),
    )


# ---------- public API ----------

def register_config_tunables():
    """
    Scan config module and register every tunable parameter
    as a UCI option.

    Call once at startup:
        register_config_tunables(UCI_TUNABLES, UCIOption)
    """
    for key in sorted(dir(config)):
        if key.startswith('_') or key in _SKIP:
            continue
        val = getattr(config, key)
        if callable(val) or isinstance(val, type):
            continue
        if not isinstance(val, (int, float, bool, str, list)):
            continue
        if key in UCI_TUNABLES:          # already registered manually
            continue

        if key in _LIST_PARAMS:
            _register_string(UCI_TUNABLES, UCIOption, key, str(val))
        elif key in _FLOAT_PARAMS or isinstance(val, float):
            _register_string(UCI_TUNABLES, UCIOption, key, str(val))
        elif isinstance(val, bool):
            _register_check(UCI_TUNABLES, UCIOption, key, val)
        elif isinstance(val, int):
            _register_spin(UCI_TUNABLES, UCIOption, key, val)
        elif isinstance(val, str):
            _register_string(UCI_TUNABLES, UCIOption, key, val)

def print_uci_options():
    for opt in UCI_TUNABLES.values():
        print(opt.uci_declaration(), flush=True)


def apply_uci_option(name: str, value: str) -> bool:
    opt = UCI_TUNABLES.get(name)
    if not opt:
        return False

    parsed_value = opt.parse_value(value)
    if opt.apply:
        opt.apply(parsed_value)

    #if name == "MULTI_CORE_BLAS":
     #   configure_multi_core_blas() # MULTI_CORE_BLAS needs special handling
    #elif name == "NN_TYPE":
     #   configure_nn_type()

    return True
