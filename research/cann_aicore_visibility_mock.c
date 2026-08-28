/*
 * Mock backend for research/cann_aicore_visibility.py --selftest.
 *
 * One shared object exporting BOTH the libascendcl subset and the libopapi
 * subset the probe binds, backed by a plain host-memory model:
 *
 *   - every aclrtMalloc / aclrtMallocHost allocation is registered host
 *     memory, so memcpys are memmove and the aclnnAdd "kernel" executes
 *     synchronously against the LANDED state of device memory;
 *   - ACLMOCK_LANDING=immediate : aclrtMemcpyAsync lands instantly (clean run)
 *   - ACLMOCK_LANDING=lag1      : at most ONE async copy stays pending; it
 *     lands when the NEXT async copy is issued or a synchronize runs. The
 *     AI-core consumer therefore reads the previous step's value (lag=1,
 *     first step reads the sentinel) -- a deterministic model of the
 *     visibility window, for verifying the probe's verdict branches;
 *   - ACLMOCK_SYNC_BLIND=1  : aclrtSynchronizeStream returns WITHOUT landing
 *     the pending copy (models "sync does not carry visibility");
 *   - ACLMOCK_EVENT_BLIND=1 : aclrtSynchronizeEvent returns WITHOUT landing
 *     (models fence-blind events).
 *
 * Single-threaded by design: the selftest exercises --dispatch direct only.
 * Build: gcc -shared -fPIC -O2 -o mock_cann.so cann_aicore_visibility_mock.c
 */
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ACL_SUCCESS 0
#define ACL_MEMCPY_HOST_TO_DEVICE 1
#define ACL_MEMCPY_DEVICE_TO_HOST 2

typedef struct Alloc {
    void *ptr;
    size_t size;
    struct Alloc *next;
} Alloc;

static Alloc *g_allocs = NULL;

typedef struct Pending {
    void *dst;
    const void *src;
    size_t n;
    int live;
} Pending;

static Pending g_pending = {NULL, NULL, 0, 0};

static int env_flag(const char *name, const char *value)
{
    const char *v = getenv(name);
    return v && strcmp(v, value) == 0;
}

static void register_alloc(void *ptr, size_t size)
{
    Alloc *a = (Alloc *)calloc(1, sizeof(Alloc));
    a->ptr = ptr;
    a->size = size;
    a->next = g_allocs;
    g_allocs = a;
}

static Alloc *find_alloc(const void *ptr)
{
    for (Alloc *a = g_allocs; a; a = a->next) {
        if (a->ptr == ptr) {
            return a;
        }
    }
    return NULL;
}

static void apply_pending(void)
{
    if (g_pending.live) {
        memmove(g_pending.dst, g_pending.src, g_pending.n);
        g_pending.live = 0;
    }
}

/* ---------------- libascendcl subset ---------------- */

int aclrtSetDevice(int32_t device) { (void)device; return ACL_SUCCESS; }
int aclrtResetDevice(int32_t device) { (void)device; return ACL_SUCCESS; }

int aclrtCreateStream(void **stream)
{
    *stream = (void *)0x51;
    return ACL_SUCCESS;
}
int aclrtDestroyStream(void *stream) { (void)stream; return ACL_SUCCESS; }
int aclrtCreateEvent(void **event)
{
    *event = (void *)0x45;
    return ACL_SUCCESS;
}
int aclrtDestroyEvent(void *event) { (void)event; return ACL_SUCCESS; }
int aclrtRecordEvent(void *event, void *stream)
{
    (void)event;
    (void)stream;
    return ACL_SUCCESS;
}
int aclrtSynchronizeEvent(void *event)
{
    (void)event;
    if (!env_flag("ACLMOCK_EVENT_BLIND", "1")) {
        apply_pending();
    }
    return ACL_SUCCESS;
}
int aclrtStreamWaitEvent(void *stream, void *event)
{
    (void)stream;
    (void)event; /* device-side ordering: never lands pending (matches the
                    fence-blind model; ordering itself is assumed) */
    return ACL_SUCCESS;
}
int aclrtSynchronizeStream(void *stream)
{
    (void)stream;
    if (!env_flag("ACLMOCK_SYNC_BLIND", "1")) {
        apply_pending();
    }
    return ACL_SUCCESS;
}

int aclrtMalloc(void **ptr, uint64_t size, int32_t policy)
{
    (void)policy;
    *ptr = malloc((size_t)size);
    if (!*ptr) {
        return 107000;
    }
    memset(*ptr, 0, (size_t)size);
    register_alloc(*ptr, (size_t)size);
    return ACL_SUCCESS;
}
int aclrtFree(void *ptr)
{
    Alloc *prev = NULL;
    for (Alloc *a = g_allocs; a; prev = a, a = a->next) {
        if (a->ptr == ptr) {
            if (prev) {
                prev->next = a->next;
            } else {
                g_allocs = a->next;
            }
            free(a);
            free(ptr);
            return ACL_SUCCESS;
        }
    }
    return 100000; /* invalid ptr */
}
int aclrtMallocHost(void **ptr, uint64_t size)
{
    *ptr = malloc((size_t)size);
    if (!*ptr) {
        return 107000;
    }
    memset(*ptr, 0, (size_t)size);
    register_alloc(*ptr, (size_t)size);
    return ACL_SUCCESS;
}
int aclrtFreeHost(void *ptr) { return aclrtFree(ptr); }

int aclrtMemcpy(void *dst, size_t dp, const void *src, size_t sp, int32_t kind)
{
    (void)dp;
    (void)sp;
    (void)kind;
    apply_pending(); /* synchronous copy semantics: everything landed first */
    memmove(dst, src, sp);
    return ACL_SUCCESS;
}

int aclrtMemcpyAsync(void *dst, size_t dp, const void *src, size_t sp, int32_t kind, void *stream)
{
    (void)dp;
    (void)kind;
    (void)stream;
    if (!find_alloc(dst) || !find_alloc(src)) {
        return 100000;
    }
    if (env_flag("ACLMOCK_LANDING", "lag1")) {
        apply_pending();
        g_pending.dst = dst;
        g_pending.src = src;
        g_pending.n = sp;
        g_pending.live = 1;
    } else {
        memmove(dst, src, sp);
    }
    return ACL_SUCCESS;
}

/* ---------------- libopapi subset ---------------- */

typedef struct {
    int32_t dtype;
    int64_t n;
    const void *dev_ptr;
} MockTensor;

typedef struct {
    int32_t dtype;
    int64_t value;
} MockScalar;

typedef struct {
    const int32_t *src;
    const int32_t *other;
    int64_t alpha;
    int64_t n;
    int32_t *out;
} MockExec;

void *aclCreateTensor(const int64_t *dims, uint64_t dim_num, int32_t dtype,
                      const int64_t *strides, int64_t offset, int32_t format,
                      const int64_t *storage_dims, uint64_t storage_dim_num, void *tensor_data)
{
    (void)strides;
    (void)offset;
    (void)format;
    (void)storage_dims;
    (void)storage_dim_num;
    if (dim_num != 1 || dims[0] <= 0) {
        return NULL;
    }
    MockTensor *t = (MockTensor *)calloc(1, sizeof(MockTensor));
    t->dtype = dtype;
    t->n = dims[0];
    t->dev_ptr = tensor_data;
    return t;
}

int aclDestroyTensor(const void *tensor)
{
    free((void *)tensor);
    return ACL_SUCCESS;
}

void *aclCreateScalar(void *value, int32_t dtype)
{
    MockScalar *s = (MockScalar *)calloc(1, sizeof(MockScalar));
    s->dtype = dtype;
    if (dtype == 9) { /* ACL_DT_INT64 */
        s->value = *(const int64_t *)value;
    } else if (dtype == 3) { /* ACL_DT_INT32 */
        s->value = *(const int32_t *)value;
    } else {
        s->value = 0;
    }
    return s;
}

int aclDestroyScalar(const void *scalar)
{
    free((void *)scalar);
    return ACL_SUCCESS;
}

/* out = self + alpha * other (int32 elementwise), executed against the
 * LANDED memory state -- the whole point of the model: whether the pending
 * tested copy has landed decides what the "kernel" observes. */
static void exec_add(MockExec *e)
{
    for (int64_t i = 0; i < e->n; i++) {
        e->out[i] = e->src[i] + (int32_t)(e->alpha * e->other[i]);
    }
}

int aclnnAddGetWorkspaceSize(const void *self, const void *other, const void *alpha,
                             const void *out, uint64_t *workspace_size, void **executor)
{
    const MockTensor *ts = (const MockTensor *)self;
    const MockTensor *to = (const MockTensor *)other;
    const MockScalar *ta = (const MockScalar *)alpha;
    const MockTensor *tout = (const MockTensor *)out;
    if (!ts || !to || !ta || !tout) {
        return 1;
    }
    if (ts->dtype != 3 || to->dtype != 3 || tout->dtype != 3) {
        return 2; /* probe uses int32 only */
    }
    MockExec *e = (MockExec *)calloc(1, sizeof(MockExec));
    e->src = (const int32_t *)ts->dev_ptr;
    e->other = (const int32_t *)to->dev_ptr;
    e->alpha = ta->value;
    e->n = ts->n;
    e->out = (int32_t *)tout->dev_ptr;
    *workspace_size = 0;
    *executor = e;
    return ACL_SUCCESS;
}

int aclnnAdd(void *workspace, uint64_t workspace_size, void *executor, void *stream)
{
    (void)workspace;
    (void)workspace_size;
    (void)stream;
    if (!executor) {
        return 1;
    }
    exec_add((MockExec *)executor);
    free(executor);
    return ACL_SUCCESS;
}
