# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research patch: static KV compaction scheduler-side hooks.

Imported conditionally by ``vllm_ascend/patch/platform/__init__.py`` when
``VLLM_ASCEND_STATIC_KV_COMPACT=1`` (default off). Wraps
``Scheduler.update_from_output`` (compaction decision, post-execution and
strictly before the next schedule in sync mode) and ``KVCacheManager.free``
(record cleanup on finish/preempt). All logic lives in
``vllm_ascend.worker.static_kv_compact``; this file only installs wrappers.
"""

from functools import wraps

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.worker import static_kv_compact

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    _original_update_from_output = Scheduler.update_from_output

    @wraps(_original_update_from_output)
    def _patched_update_from_output(self, scheduler_output, model_runner_output):
        static_kv_compact.maybe_compact_batch(self, scheduler_output)
        return _original_update_from_output(self, scheduler_output, model_runner_output)

    Scheduler.update_from_output = _patched_update_from_output

    _original_free = KVCacheManager.free

    @wraps(_original_free)
    def _patched_free(self, request):
        static_kv_compact.forget(request.request_id)
        return _original_free(self, request)

    KVCacheManager.free = _patched_free


install()
