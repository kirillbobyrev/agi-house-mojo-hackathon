"""
Benchmark the prefix sum kernel.
"""

from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    BenchConfig,
)
from gpu import global_idx
from gpu.host import DeviceBuffer, DeviceContext
from math import ceildiv, floor
from memory import stack_allocation, UnsafePointer
from testing import assert_equal
from sys import sizeof
from gpu.memory import load
from prefix_sum import prefix_sum_naive


fn pretty_print_float(val: Float64) -> String:
    """This converts the float value to a string, but omits the fractional part
    if not needed (e.g. prints 2 instead of 2.0).
    """
    if Float64(floor(val)) == val:
        return String(Int(val))
    return String(val)


fn human_memory(size: Int) -> String:
    alias KB = 1024
    alias MB = KB * KB
    alias GB = MB * KB

    if size >= GB:
        return pretty_print_float(Float64(size) / GB) + "GiB"

    if size >= MB:
        return pretty_print_float(Float64(size) / MB) + "MiB"

    if size >= KB:
        return pretty_print_float(Float64(size) / KB) + "KiB"

    return String(size) + "B"


def run_benchmark[
    type: DType, block_size: Int
](num_elements: Int, ctx: DeviceContext, mut bench_manager: Bench):
    output = ctx.enqueue_create_buffer[type](num_elements).enqueue_fill(0)
    input = ctx.enqueue_create_buffer[type](num_elements).enqueue_fill(0)

    var grid_dim = ceildiv(num_elements, block_size)

    block_data = ctx.enqueue_create_buffer[type](grid_dim).enqueue_fill(0)
    block_counter = ctx.enqueue_create_buffer[DType.uint64](1).enqueue_fill(0)

    with input.map_to_host() as input_host:
        for i in range(num_elements):
            input_host[i] = i

    @always_inline
    @parameter
    fn run_func() raises:
        ctx.enqueue_function[
            prefix_sum_naive[type=type, exclusive=False, block_size=block_size]
        ](
            input.unsafe_ptr(),
            output.unsafe_ptr(),
            block_data.unsafe_ptr(),
            num_elements,
            block_counter,
            grid_dim=grid_dim,
            block_dim=block_size,
        )

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            run_func()

        b.iter_custom[kernel_launch](ctx)

    var num_bytes = 2 * num_elements * sizeof[type]()
    bench_manager.bench_function[bench_func](
        BenchId(
            "stream",
            input_id=String(
                "length=",
                human_memory(num_bytes),
                "/",
                "type=",
                type,
                "/",
                "block_dim=",
                block_size,
            ),
        ),
        ThroughputMeasure(BenchMetric.bytes, num_bytes),
    )

    ctx.synchronize()

    with output.map_to_host() as out_host:
        for i in range(num_elements):
            assert_equal(
                out_host[i], i, "Output mismatch at index " + String(i)
            )


def main():
    var bench_manager = Bench(BenchConfig(max_iters=1))

    with DeviceContext() as ctx:
        print("Running on device:", ctx.name())

        alias NUM_ELEMENTS = 1024 * 1024 * 128

        alias TYPE = DType.uint32
        alias BLOCK_SIZE = 128

        run_benchmark[TYPE, BLOCK_SIZE](NUM_ELEMENTS, ctx, bench_manager)

    bench_manager.dump_report()
