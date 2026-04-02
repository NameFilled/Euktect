# flash_attn_cpu/utils/distributed.py
"""CPU-compatible distributed utils stubs."""


def sync_shared_params(module, process_group):
    """No-op: no distributed sync needed for CPU-only single-process inference."""
    pass


def all_gather_raw(input_, process_group, async_op=False):
    """No-op: returns input unchanged for CPU-only single-process inference."""
    return input_, None
