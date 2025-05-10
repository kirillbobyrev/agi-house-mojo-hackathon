"""
This is a benchmark for Speed of Light for the Prefix Sum algorithm, it measures
the throughput of the theoretical maximum speed of the algorithm execution.
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
from layout import Layout, LayoutTensor
from testing import assert_equal
from sys import sizeof
from gpu.memory import load

alias NUM_ELEMENTS = 1024 * 1024 * 128
alias BLOCK_SIZE = 128
alias TYPE = DType.uint32
alias SIMD_WIDTH = 8


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


fn stream_kernel_layout[
    type: DType, width: Int, block_size: Int, layout: Layout
](
    input: LayoutTensor[mut=False, type, layout],
    output: LayoutTensor[mut=True, type, layout],
    size: Int,
):
    var idx = global_idx.x
    if idx >= size:
        return

    var val: SIMD[type, width]

    input_view = input.vectorize[width]()
    output_view = output.vectorize[width]()

    val = rebind[SIMD[type, width]](input_view[idx])

    output_view.store[width](idx, 0, val)


fn stream_kernel_ptr[
    type: DType, width: Int, block_size: Int
](
    input: UnsafePointer[SIMD[type, width]],
    output: UnsafePointer[SIMD[type, width]],
    size: Int,
):
    var idx = global_idx.x
    if idx >= size:
        return

    output[idx] = input[idx]


def run_benchmark[
    type: DType, width: Int, block_size: Int
](num_elements: Int, ctx: DeviceContext, mut bench_manager: Bench):
    input = ctx.enqueue_create_buffer[type](num_elements).enqueue_fill(0)
    output = ctx.enqueue_create_buffer[type](num_elements).enqueue_fill(0)

    with input.map_to_host() as input_host:
        for i in range(num_elements):
            input_host[i] = i

    alias layout = Layout.row_major(NUM_ELEMENTS)

    input_tensor = LayoutTensor[type, layout](input.unsafe_ptr())
    output_tensor = LayoutTensor[type, layout](input.unsafe_ptr())

    var grid_dim = ceildiv(num_elements, block_size)

    @parameter
    fn run_func() raises:
        ctx.enqueue_function[
            stream_kernel_layout[type, width, block_size, layout]
        ](
            input_tensor,
            output_tensor,
            ceildiv(num_elements, width),
            grid_dim=grid_dim,
            block_dim=block_size,
        )
        """
        ctx.enqueue_function[stream_kernel_ptr[type, width, block_size]](
            input,
            output,
            ceildiv(num_elements, width),
            grid_dim=grid_dim,
            block_dim=block_size,
        )
        """

    @parameter
    fn bench_func(mut b: Bencher):
        @parameter
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
                "width=",
                width,
                "/",
                "block_dim=",
                block_size,
            ),
        ),
        ThroughputMeasure(BenchMetric.bytes, num_bytes),
    )

    '''
    with output.map_to_host() as out_host:
        print(out_host)
        for i in range(num_elements):
            assert_equal(
                out_host[i], i, "Output mismatch at index " + String(i)
            )
    '''


def main():
    var bench_manager = Bench(BenchConfig(max_iters=1))

    with DeviceContext() as ctx:
        print("Running on device:", ctx.name())

        run_benchmark[TYPE, SIMD_WIDTH, BLOCK_SIZE](
            NUM_ELEMENTS, ctx, bench_manager
        )

    bench_manager.dump_report()
