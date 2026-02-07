"""
uci_config_bridge.py

Auto-registers config.py parameters as UCI options so that cutechess-cli
can configure each engine instance via  option.PARAM=value .

Integration (add to uci.py after defining UCI_TUNABLES):
    from uci_config_bridge import register_config_tunables
    register_config_tunables(UCI_TUNABLES, UCIOption)
"""
import os

import config

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

def register_config_tunables(uci_tunables_dict, UCIOptionClass):
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
        if key in uci_tunables_dict:          # already registered manually
            continue

        if key in _LIST_PARAMS:
            _register_string(uci_tunables_dict, UCIOptionClass, key, str(val))
        elif key in _FLOAT_PARAMS or isinstance(val, float):
            _register_string(uci_tunables_dict, UCIOptionClass, key, str(val))
        elif isinstance(val, bool):
            _register_check(uci_tunables_dict, UCIOptionClass, key, val)
        elif isinstance(val, int):
            _register_spin(uci_tunables_dict, UCIOptionClass, key, val)
        elif isinstance(val, str):
            _register_string(uci_tunables_dict, UCIOptionClass, key, val)