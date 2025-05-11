from gpu import block, warp, lane_id, warp_id
from gpu import id
from gpu import thread_idx, block_dim, global_idx, grid_dim
from gpu.host import DeviceContext
from sys import sizeof
from gpu.intrinsics import threadfence
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import log2, ceildiv
from memory import stack_allocation, UnsafePointer
from os.atomic import Atomic
from testing import assert_equal
from time import perf_counter_ns
from gpu.sync import barrier
from time import sleep
import gpu.globals
import time

# ustdlib (patched stdlib functions)
import common

alias BLOCK_DIM = 256
alias SIZE = 2**30
alias GRID_DIM = ceildiv(SIZE, BLOCK_DIM)
alias TYPE = DType.uint32
alias EXCLUSIVE = False
alias GPU_ID = 1
alias TEST = False
alias LAYOUT = Layout.row_major(SIZE)
alias WIDTH = 8


# Constants
alias BLOCK_COMPLETE = 1 << 1
alias PREFIX_COMPLETE = 1 << 0


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
    var val = input[global_tid] if global_tid < size else 0

    # Step 1: Upsweep: sum the values in each block.
    var per_block_sum = common.block_sum[
        type=type,
        block_size=block_size,
    ](val)

    if thread_idx.x == 0:
        # Store the block sum in shared memory.
        var aggregate_value = per_block_sum << 1
        if block_id == 0:
            aggregate_value |= PREFIX_COMPLETE
        _ = Atomic.store(block_data + block_id, aggregate_value)

    # Step 2: Propagate the block sum to all other threadds in the grid.
    # This a very naive implementation of the lookback, which requires the
    # previous block to be complete and causes a lot of problems with the memory
    # contention (all blocks are polling the same memory location).
    if thread_idx.x == 0:
        if block_id != 0:
            while True:
                # Load the previous block's aggregate value.
                var previous_block_aggregate = Atomic.fetch_add(
                    block_data + block_id - 1, 0
                )
                if previous_block_aggregate & PREFIX_COMPLETE:
                    complete_block_prefix = previous_block_aggregate >> 1
                    _ = Atomic.store(
                        block_data + block_id,
                        ((per_block_sum + complete_block_prefix) << 1)
                        | PREFIX_COMPLETE,
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

    if global_tid < size:
        output[global_tid] = prefix_sum


fn prefix_sum_decoupled_lookback[
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
    var val = input[global_tid] if global_tid < size else 0

    # Step 1: Upsweep: sum the values in each block.
    var per_block_sum = common.block_sum[
        type=type,
        block_size=block_size,
    ](val)

    if thread_idx.x == 0:
        # Store the block sum in shared memory.
        var aggregate_value = per_block_sum << 2
        if block_id == 0:
            aggregate_value |= PREFIX_COMPLETE
        else:
            aggregate_value |= BLOCK_COMPLETE
        # print("block_id:", block_id, "aggregate_value:", aggregate_value)
        _ = Atomic.store(block_data + block_id, aggregate_value)

    # Step 2: Propagate the block sum to all other threadds in the grid.
    # This is the core of the algorithm, the decoupled look-back operation.

    # Aggregates the sum of all previous blocks. Each block requires this
    # to calculate the final value of the prefix sum.
    var block_prefix: Scalar[type] = 0

    # Only the leaders participate in the lookback operation.
    if thread_idx.x == 0:
        # The first block doesn't need the prefix, it's already complete.
        if block_id != 0:
            # Keep track of the previous incomplete block to poll for
            # completion.
            var polling_block_id = rebind[Int](block_id - 1)

            while True:
                # Poll the value.
                var previous_block_value = block_data[polling_block_id]

                # If the previous block has complete prefix, this is all current
                # thread needs. Add it to the block_prefix and break.
                if previous_block_value & PREFIX_COMPLETE:
                    block_prefix += rebind[Scalar[type]](
                        previous_block_value >> 2
                    )
                    break
                # If the previus block has completed it's partial sum, but
                # doesn't have the prefix, we should add it to the block_prefix
                # and continue polling its predecessors.
                elif previous_block_value & BLOCK_COMPLETE:
                    block_prefix += rebind[Scalar[type]](
                        previous_block_value >> 2
                    )
                    polling_block_id -= 1
                else:
                    # Do we want to to sleep here?
                    continue

            # Mark the block value as complete for the successors.
            _ = Atomic.store(
                block_data + block_id,
                ((per_block_sum + block_prefix) << 2) | PREFIX_COMPLETE,
            )

    # Step 3: Downsweep: calculate the prefix sum in each block.
    var prefix_sum = common.block_prefix_sum[
        type=type,
        block_size=block_size,
        exclusive=exclusive,
    ](val)

    # Leader has received its block prefix, but the followers have not.
    # Broadcast from the leader to all threads in the block.
    addend = block.broadcast[block_size=block_size](block_prefix, 0)

    if global_tid < size:
        output[global_tid] = prefix_sum + addend


fn prefix_sum_decoupled_lookback_vectorized[
    type: DType,
    width: Int,
    block_size: Int,
    layout: Layout,
    exclusive: Bool = True,
](
    input: LayoutTensor[mut=False, type, layout],
    output: LayoutTensor[mut=True, type, layout],
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
    var block_id = block.broadcast[type = DType.uint64, block_size=block_size](
        counter
    )

    var global_tid = rebind[Int](block_id * block_size + thread_idx.x)

    # print(
    #     "grid_dim", grid_dim.x, "block_id", block_id, "thread_idx", thread_idx.x, "size", size, "global_tid", global_tid
    # )

    # Load the vectorized value from the input.
    var input_view = input.vectorize[width]()
    var val: SIMD[type, width] = input_view.load[width](
        global_tid, 0
    ) if global_tid < size else 0

    # Step 1: Upsweep: sum the values in each block.
    var vector_sum: Scalar[type] = val.reduce_add()
    var per_block_sum: Scalar[type] = common.block_sum[
        type=type,
        block_size=block_size,
    ](vector_sum)

    if thread_idx.x == 0:
        # Store the block sum in shared memory.
        var aggregate_value = per_block_sum << 2
        if block_id == 0:
            aggregate_value |= PREFIX_COMPLETE
        else:
            aggregate_value |= BLOCK_COMPLETE
        _ = Atomic.store(block_data + block_id, aggregate_value)

    # Step 2: Propagate the block sum to all other threadds in the grid.
    # This is the core of the algorithm, the decoupled look-back operation.

    # Aggregates the sum of all previous blocks. Each block requires this
    # to calculate the final value of the prefix sum.
    var block_prefix: Scalar[type] = 0

    # Only the leaders participate in the lookback operation.
    if thread_idx.x == 0:
        # The first block doesn't need the prefix, it's already complete.
        if block_id != 0:
            # Keep track of the previous incomplete block to poll for
            # completion.
            var polling_block_id = rebind[Int](block_id - 1)

            while True:
                # Poll the value.
                var previous_block_value = block_data[polling_block_id]

                # If the previous block has complete prefix, this is all current
                # thread needs. Add it to the block_prefix and break.
                if previous_block_value & PREFIX_COMPLETE:
                    block_prefix += rebind[Scalar[type]](
                        previous_block_value >> 2
                    )
                    break
                # If the previus block has completed it's partial sum, but
                # doesn't have the prefix, we should add it to the block_prefix
                # and continue polling its predecessors.
                elif previous_block_value & BLOCK_COMPLETE:
                    block_prefix += rebind[Scalar[type]](
                        previous_block_value >> 2
                    )
                    polling_block_id -= 1
                else:
                    # Do we want to to sleep here?
                    continue

            # Mark the block value as complete for the successors.
            _ = Atomic.store(
                block_data + block_id,
                ((per_block_sum + block_prefix) << 2) | PREFIX_COMPLETE,
            )

    # Step 3: Downsweep: calculate the prefix sum in each block.

    # Perform the prefix scan on individual vector sums.
    var simd_prefix_sum = common.block_prefix_sum[
        type=type, block_size=block_size, exclusive=True
    ](val.reduce_add())

    var prefix_sum: SIMD[type, width] = 0

    var running_sum: Scalar[type] = 0

    @parameter
    for i in range(width):
        running_sum += val[i]
        prefix_sum[i] += running_sum

    prefix_sum += simd_prefix_sum

    addend = block.broadcast[block_size=block_size](block_prefix, 0)
    prefix_sum += addend

    if global_tid < size:
        output.store[width](global_tid * width, 0, prefix_sum)


def main():
    with DeviceContext(GPU_ID) as ctx:
        print("device:", ctx.name(), "id:", ctx.id())

        print("Parameters")
        print(
            "array_size:",
            common.human_memory(SIZE * sizeof[TYPE]()),
            "block_size:",
            BLOCK_DIM,
            "simd_width:",
            WIDTH,
            "blocks_count:",
            GRID_DIM,
            "dtype:",
            TYPE,
            "testing:",
            TEST,
        )

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

        var current_time = perf_counter_ns()

        input_tensor = LayoutTensor[TYPE, LAYOUT](input.unsafe_ptr())
        output_tensor = LayoutTensor[TYPE, LAYOUT](output.unsafe_ptr())

        """
        ctx.enqueue_function[
            prefix_sum_decoupled_lookback[
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
        """
        print("passing size", SIZE // WIDTH)
        ctx.enqueue_function[
            prefix_sum_decoupled_lookback_vectorized[
                type=TYPE,
                width=WIDTH,
                exclusive=EXCLUSIVE,
                block_size=BLOCK_DIM,
                layout=LAYOUT,
            ]
        ](
            input_tensor,
            output_tensor,
            block_data.unsafe_ptr(),
            SIZE // WIDTH,
            block_counter,
            grid_dim=GRID_DIM,
            block_dim=BLOCK_DIM,
        )

        ctx.synchronize()

        var elapsed_time_ms = (perf_counter_ns() - current_time) / 1e6
        print("Elapsed time:", elapsed_time_ms, "(ms)")

        @parameter
        if TEST:
            with output.map_to_host() as out_host:
                print("out:", out_host)
                print("block_data:", block_data)

                for i in range(SIZE):

                    @parameter
                    if EXCLUSIVE:
                        assert_equal(out_host[i], i, "at i: " + String(i))
                    else:
                        assert_equal(out_host[i], i + 1, "at i: " + String(i))
