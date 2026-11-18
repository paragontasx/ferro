#!/usr/bin/env python3
"""OpenCL device bandwidth probe for the ferro iGPU (Arrow Lake-H 8086:7D67).

ctypes against the custom ICD shim. Bitfield params are 64-bit (cl_bitfield) —
all argtypes are declared explicitly to avoid the classic ctypes ABI bug.
Run with the same env the ferro GPU path uses.
"""
import ctypes as C

LIB = "/home/hivemind/.local/ferro-ocl/shim/libOpenCL.so.1"
cl = C.CDLL(LIB)

CL_FALSE, CL_TRUE = 0, 1
CL_DEVICE_MAX_COMPUTE_UNITS, CL_DEVICE_MAX_CLOCK_FREQ, CL_DEVICE_NAME = 0x1004, 0x1005, 0x100B
CL_DEVICE_LOCAL_MEM_SIZE, CL_DEVICE_MAX_MEM_ALLOC, CL_DEVICE_MAX_WORK_GROUP = 0x1003, 0x1002, 0x1006
CL_DEVICE_MAX_WORK_GROUP_SZ, CL_DEVICE_MAX_WORK_ITEM_DIMS, CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1006, 0x1000, 0x1001
CL_DEVICE_TYPE_GPU = C.c_uint64(1 << 2)
CL_CONTEXT_PLATFORM = 0x1084
CL_QUEUE_PROFILING_ENABLE = C.c_uint64(1 << 0)
CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END = 0x1284, 0x1285
CL_MEM_READ_ONLY, CL_MEM_READ_WRITE = C.c_uint64(1 << 2), C.c_uint64(1 << 0)

for _f in ("clCreateContext", "clCreateCommandQueue", "clCreateBuffer", "clCreateProgramWithSource", "clCreateKernel"):
    getattr(cl, _f).restype = C.c_void_p

def chk(rc, what=""):
    v = rc.value if isinstance(rc, C._SimpleCData) else rc
    if v != 0:
        raise RuntimeError(f"{what} cl error {v:#x}")

cl.clGetPlatformIDs.argtypes = [C.c_uint32, C.POINTER(C.c_void_p), C.POINTER(C.c_uint32)]
cl.clGetDeviceIDs.argtypes = [C.c_void_p, C.c_uint64, C.c_uint32, C.POINTER(C.c_void_p), C.POINTER(C.c_uint32)]
cl.clGetDeviceInfo.argtypes = [C.c_void_p, C.c_uint32, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
cl.clGetPlatformInfo.argtypes = [C.c_void_p, C.c_uint32, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
cl.clCreateContext.argtypes = [C.POINTER(C.c_int), C.c_uint32, C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]
cl.clCreateCommandQueueWithProperties.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_int)]
cl.clCreateBuffer.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.c_void_p, C.POINTER(C.c_int)]
cl.clEnqueueCopyBuffer.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t, C.c_size_t,
                                   C.c_uint32, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
cl.clFinish.argtypes = [C.c_void_p]
cl.clGetEventProfilingInfo.argtypes = [C.c_void_p, C.c_uint32, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
cl.clReleaseMemObject.argtypes = [C.c_void_p]
cl.clReleaseCommandQueue.argtypes = [C.c_void_p]
cl.clReleaseContext.argtypes = [C.c_void_p]

def info_str(obj, param, platform=False):
    buf = C.create_string_buffer(512)
    sz = C.c_size_t(0)
    fn = cl.clGetPlatformInfo if platform else cl.clGetDeviceInfo
    rc = fn(obj, param, 512, buf, C.byref(sz))
    if rc != 0:
        return f"(devinfo rc={rc:#x})"
    return buf.value.decode()

def info_int(obj, param):
    v = C.c_int(0); sz = C.c_size_t(0)
    rc = cl.clGetDeviceInfo(obj, param, 4, C.byref(v), C.byref(sz))
    if rc != 0:
        raise RuntimeError(f"devinfo int rc={rc:#x}")
    return v.value

n = C.c_uint32(0); ids = (C.c_void_p * 1)()
chk(cl.clGetPlatformIDs(1, ids, C.byref(n)), "plat")
plat = ids[0]
print("platform:", info_str(plat, 0x0901, platform=True))

dev_a = (C.c_void_p * 1)(); nd = C.c_uint32(0)
chk(cl.clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, dev_a, C.byref(nd)), "dev")
dev = dev_a[0]
devp = C.c_void_p(dev)
try:
    print("device:", info_str(dev, CL_DEVICE_NAME),
          "| compute units:", info_int(dev, CL_DEVICE_MAX_COMPUTE_UNITS),
          "| clock:", info_int(dev, CL_DEVICE_MAX_CLOCK_FREQ), "MHz")
except RuntimeError as e:
    print("device info unavailable:", e)

err = C.c_int(0)
print(f"plat={plat:#x} dev={dev:#x}")
ctx = cl.clCreateContext(None, 1, C.byref(devp), None, None, C.byref(err))
chk(err, "ctx")
qprops = (C.c_uint64 * 3)(CL_QUEUE_PROFILING_ENABLE.value, 0, 0)
cl.clCreateCommandQueue.argtypes = [C.c_void_p, C.c_void_p, C.c_uint64, C.POINTER(C.c_int)]
q = cl.clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, C.byref(err))
chk(err, "queue")

def mkbuf(size, flags):
    b = cl.clCreateBuffer(ctx, flags, size, None, C.byref(err))
    chk(err, "buf"); return b

import time as T
def prof(event=None):  # wall-clock fallback: shim/CR do not expose event profiling
    return None

def bench_copy(label, size, iters, a=None, b=None):
    ts = []
    for _ in range(2):
        cl.clEnqueueCopyBuffer(q, a, b, 0, 0, size, 0, None, None)
    cl.clFinish(q)
    for _ in range(iters):
        t0 = T.perf_counter()
        cl.clEnqueueCopyBuffer(q, a, b, 0, 0, size, 0, None, None)
        cl.clFinish(q)
        ts.append(T.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts)//2]
    if size >= (1 << 20):
        print(f"{label}: median {med*1e3:.3f} ms  copy {size/med/1e9:.2f} GB/s  bus {2*size/med/1e9:.2f} GB/s")
    else:
        print(f"{label}: median {med*1e3:.4f} ms  ({size} B)")

SIZE = 512 * (1 << 20)
a = C.c_void_p(mkbuf(SIZE, CL_MEM_READ_ONLY))
b = C.c_void_p(mkbuf(SIZE, CL_MEM_READ_WRITE))
bench_copy("copy 512MB", SIZE, 30, a, b)
bench_copy("copy 512MB again", SIZE, 20, a, b)
for sz, it in ((64*(1<<20), 40), (16*(1<<20), 60), (1*(1<<20), 100), (256*1024, 150), (32*1024, 200)):
    bench_copy(f"copy {sz//(1<<20)}M" if sz >= (1<<20) else f"copy {sz//1024}K", sz, it, a, b)

cl.clEnqueueCopyBuffer(q, a, b, 0, 0, 32*1024, 0, None, None)
t0 = T.perf_counter(); cl.clFinish(q)
print(f"cold Finish round-trip: {(T.perf_counter()-t0)*1e3:.3f} ms")

# ---- kernel-based read/write rates ----
KSRC = r"""
__kernel void k_read_only(__global const float* in, __global float* out, const int n) {
    int gid = get_global_id(0);
    float v = 0.0f;
    for (int i = 0; i < 4; ++i) v += in[4*gid + i];
    if (gid == 0) out[0] = v;   // keep the loads alive
}
__kernel void k_write_only(__global float* out, const int n) {
    int gid = get_global_id(0);
    for (int i = 0; i < 4; ++i) out[4*gid + i] = 1.0f;
}
__kernel void k_read_write(__global const float* in, __global float* out, const int n) {
    int gid = get_global_id(0);
    for (int i = 0; i < 4; ++i) out[4*gid + i] = in[4*gid + i];
}
"""
cl.clCreateProgramWithSource.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(C.c_char_p), C.POINTER(C.c_size_t), C.POINTER(C.c_int)]
cl.clBuildProgram.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(C.c_void_p), C.c_char_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]
cl.clCreateKernel.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int)]
cl.clSetKernelArg.argtypes = [C.c_void_p, C.c_uint32, C.c_size_t, C.c_void_p]
cl.clEnqueueNDRangeKernel.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t), C.POINTER(C.c_size_t), C.c_uint32, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
cl.clReleaseKernel.argtypes = [C.c_void_p]
cl.clReleaseProgram.argtypes = [C.c_void_p]

src = KSRC.encode()
srcarr = (C.c_char_p * 1)(src)
srcp = C.cast(srcarr, C.POINTER(C.c_char_p))
srclen = C.c_size_t(len(src))
pr = cl.clCreateProgramWithSource(ctx, 1, srcp, C.byref(srclen), C.byref(err))
chk(err, "progsrc")
rc = cl.clBuildProgram(pr, 1, C.byref(devp), b"", None, None, C.byref(err))
if rc != 0:
    # fetch build log
    cl.clGetProgramBuildInfo.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
    lg = C.create_string_buffer(4096); lsz = C.c_size_t(0)
    cl.clGetProgramBuildInfo(pr, dev, 0x1182, 4096, lg, C.byref(lsz))  # CL_PROGRAM_BUILD_LOG
    raise RuntimeError(f"build failed rc={rc:#x} err={err.value:#x}: {lg.value.decode(errors='replace')[:1200]}")
chk(rc, "build")

def kern(name, args):
    k = cl.clCreateKernel(pr, name.encode(), C.byref(err)); chk(err, "kern "+name)
    for i, (a_, sz) in enumerate(args):
        chk(cl.clSetKernelArg(k, i, sz, a_), "setarg")
    return k

def run_kernel(label, k, nbytes_traffic, iters=30):
    gsz = C.c_size_t(SIZE // (4 * 4)); lsz = C.c_size_t(64)
    n = C.c_int(SIZE // 4)
    for _ in range(2):
        chk(cl.clEnqueueNDRangeKernel(q, k, 1, None, C.byref(gsz), C.byref(lsz), 0, None, None), "enqueue")
    cl.clFinish(q)
    ts = []
    for _ in range(iters):
        t0 = T.perf_counter()
        chk(cl.clEnqueueNDRangeKernel(q, k, 1, None, C.byref(gsz), C.byref(lsz), 0, None, None), "enqueue")
        cl.clFinish(q)
        ts.append(T.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts)//2]
    print(f"{label}: median {med*1e3:.3f} ms  traffic {nbytes_traffic/med/1e9:.2f} GB/s")

# n arg buffers
nint = C.c_int(SIZE // 4)

k_r = kern("k_read_only", [(C.byref(a), C.sizeof(C.c_void_p)), (C.byref(b), C.sizeof(C.c_void_p)), (C.byref(nint), C.sizeof(C.c_int))])
run_kernel("read-only 512MB", k_r, SIZE)
k_w = kern("k_write_only", [(C.byref(b), C.sizeof(C.c_void_p)), (C.byref(nint), C.sizeof(C.c_int))])
run_kernel("write-only 512MB", k_w, SIZE)
k_rw = kern("k_read_write", [(C.byref(a), C.sizeof(C.c_void_p)), (C.byref(b), C.sizeof(C.c_void_p)), (C.byref(nint), C.sizeof(C.c_int))])
run_kernel("read+write 512MB (kernel)", k_rw, 2 * SIZE)

cl.clReleaseKernel(k_r); cl.clReleaseKernel(k_w); cl.clReleaseKernel(k_rw)
cl.clReleaseProgram(pr)

cl.clReleaseMemObject(a); cl.clReleaseMemObject(b)
cl.clReleaseCommandQueue(q); cl.clReleaseContext(ctx)
