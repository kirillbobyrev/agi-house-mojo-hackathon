from gpu import block, warp, lane_id, warp_id
from gpu import id
from gpu import thread_idx, block_dim
from gpu.host import DeviceContext
from gpu.intrinsics import threadfence
from gpu.memory import AddressSpace
from math import log2, ceildiv
from memory import stack_allocation, UnsafePointer
from os.atomic import Atomic
from testing import assert_equal
from time import sleep
import gpu.globals
import time

# ustdlib (patched stdlib functions)
import common

# Constants
alias COMPLETE = 1


fn prefix_sum_naive[
    type: DType,
    block_size: Int,
    exclusive: Bool = True,
](
    input: UnsafePointer[Scalar[type]],
    output: UnsafePointer[Scalar[type]],
    block_data: UnsafePointer[Scalar[type]],
    size: Int,
    block_counter: UnsafePointer[Scalar[DType.uint64]],
):
    # Use the atomic block counter to make sure that the blocks responsible for
    # the tiles at the beginning of the grid, so that the later blocks (that
    # depend on it) in the grid for sum propagation are not deadlocked.
    var counter: Scalar[DType.uint64] = 0
    if thread_idx.x == 0:
        counter = Atomic.fetch_add(block_counter, 1)
    block_id = block.broadcast[
        type = DType.uint64, width=1, block_size=block_size
    ](counter)

    var global_tid = block_id * block_size + thread_idx.x
    # TODO: This is not an optimized load?
    var val = input[global_tid]

    # Step 1: Upsweep: sum the values in each block.
    var per_block_sum = common.block_sum[
        type=type,
        block_size=block_size,
    ](val)

    if thread_idx.x == 0:
        # Store the block sum in shared memory.
        var aggregate_value = per_block_sum << 1
        if block_id == 0:
            aggregate_value |= COMPLETE
        block_data[block_id] = aggregate_value
        _ = Atomic.store(block_data + block_id, aggregate_value)

    # Step 2: Propagate the block sum to all other threadds in the grid.
    # This is the core of the algorithm, the decoupled look-back operation.

    current_ns = time.perf_counter_ns()

    # This is a naive unparallel propagation of the block sum.
    # TODO: Replace with actual decoupled look-back operation.
    # TODO: Even better, replace with parallel decoupled look-back operation.
    if thread_idx.x == 0:
        if block_id != 0:
            while True:
                # Load the previous block's aggregate value.
                # TODO: This is not how to load properly, I'm assuming? There
                # has to be a better way to load.
                var previous_block_aggregate = Atomic.fetch_add(
                    block_data + block_id - 1, 0
                )
                if previous_block_aggregate & COMPLETE:
                    complete_block_prefix = previous_block_aggregate >> 1
                    _ = Atomic.store(
                        block_data + block_id,
                        ((per_block_sum + complete_block_prefix) << 1)
                        | COMPLETE,
                    )
                    break

    # Step 3: Downsweep: calculate the prefix sum in each block.
    var prefix_sum = common.block_prefix_sum[
        type=type,
        block_size=block_size,
        exclusive=exclusive,
    ](val)

    # TODO: Fetch the block sum from global memory, broadcast it to all threads
    # and add it to the prefix sum.
    if block_id > 0:
        prefix_sum += block_data[block_id - 1] >> 1

    output[global_tid] = prefix_sum


def main():
    with DeviceContext() as ctx:
        print("Running on device:", ctx.name())

        alias BLOCK_DIM = 256
        alias SIZE = 2**25
        alias GRID_DIM = ceildiv(SIZE, BLOCK_DIM)
        alias TYPE = DType.uint32
        alias EXCLUSIVE = False

        output = ctx.enqueue_create_buffer[TYPE](SIZE).enqueue_fill(0)
        input = ctx.enqueue_create_buffer[TYPE](SIZE).enqueue_fill(0)
        block_data = ctx.enqueue_create_buffer[TYPE](GRID_DIM).enqueue_fill(0)

        # Fill the input buffer with values from 0 to SIZE-1.
        with input.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = 1

        block_counter = ctx.enqueue_create_buffer[DType.uint64](1).enqueue_fill(
            0
        )

        ctx.enqueue_function[
            prefix_sum_naive[
                type=TYPE, exclusive=EXCLUSIVE, block_size=BLOCK_DIM
            ]
        ](
            input.unsafe_ptr(),
            output.unsafe_ptr(),
            block_data.unsafe_ptr(),
            SIZE,
            block_counter,
            grid_dim=GRID_DIM,
            block_dim=BLOCK_DIM,
        )

        ctx.synchronize()

        with output.map_to_host() as out_host:
            print("out:", out_host)
            print("block_data:", block_data)

            for i in range(SIZE):

                @parameter
                if EXCLUSIVE:
                    assert_equal(out_host[i], i, "at i: " + String(i))
                else:
                    assert_equal(out_host[i], i + 1, "at i: " + String(i))
