from memory import stack_allocation
from gpu.id import WARP_SIZE, warp_id, lane_id
from gpu.memory import AddressSpace
from gpu.warp import (
    sum as warp_sum,
    shuffle_up as warp_shuffle_up,
    lane_group_sum,
    prefix_sum as warp_prefix_sum,
)
from math import floor
from gpu.sync import barrier
from bit import log2_floor
from gpu.block import broadcast as block_broadcast


@always_inline
fn block_reduce[
    type: DType,
    width: Int, //,
    block_size: Int,
    warp_reduce_fn: fn[dtype: DType, width: Int] (
        SIMD[dtype, width]
    ) capturing -> SIMD[dtype, width],
    broadcast: Bool = False,
](val: SIMD[type, width], *, initial_val: SIMD[type, width]) -> SIMD[
    type, width
]:
    """Performs a generic block-level reduction operation.
    This function implements a block-level reduction using warp-level operations
    and shared memory for inter-warp communication. All threads in the block
    participate to compute the final reduced value.
    Parameters:
        type: The data type of the SIMD elements.
        width: The SIMD width for vector operations.
        block_size: The number of threads in the block.
        warp_reduce_fn: A function that performs warp-level reduction.
        broadcast: If True, the final reduced value is broadcast to all
            threads in the block. If False, only the first thread will have the
            complete result.
    Args:
        val: The input value from each thread to include in the reduction.
        initial_val: The initial value for the reduction.
    Returns:
        If broadcast is True, each thread in the block will receive the reduced
        value. Otherwise, only the first thread will have the complete result.
    """
    constrained[
        block_size % WARP_SIZE == 0,
        "Block size must be a multiple of warp size",
    ]()

    # Allocate shared memory for inter-warp communication.
    alias n_warps = block_size // WARP_SIZE
    var shared_mem = stack_allocation[
        n_warps * width, type, address_space = AddressSpace.SHARED
    ]()

    # Step 1: Perform warp-level reduction.
    var warp_result = warp_reduce_fn(val)

    # Step 2: Store warp results to shared memory
    var wid = warp_id()
    var lid = lane_id()
    # Each leader thread (lane 0) is responsible for its warp.
    if lid == 0:
        shared_mem.store(wid, warp_result)

    barrier()

    # Step 3: Have the first warp reduce all warp results.
    if wid == 0:
        # Make sure that the "ghost" warps do not contribute to the sum.
        var block_val = initial_val
        # Load values from the shared memory (ith lane will have ith warp's
        # value).
        if lid < n_warps:
            block_val = shared_mem.load[width=width](lid)

        # Reduce across the first warp
        warp_result = warp_sum(block_val)

    @parameter
    if broadcast:
        # Broadcast the result to all threads in the block
        warp_result = block_broadcast[block_size=block_size](warp_result, 0)

    return warp_result


@always_inline
fn block_sum[
    type: DType, width: Int, //, *, block_size: Int, broadcast: Bool = False
](val: SIMD[type, width]) -> SIMD[type, width]:
    """Computes the sum of values across all threads in a block.
    Performs a parallel reduction using warp-level operations and shared memory
    to find the global sum across all threads in the block.
    Parameters:
        type: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.
        broadcast: If True, the final sum is broadcast to all threads in the
            block. If False, only the first thread will have the complete sum.
    Args:
        val: The SIMD value to reduce. Each thread contributes its value to the
             sum.
    Returns:
        If broadcast is True, each thread in the block will receive the final
        sum. Otherwise, only the first thread will have the complete sum.
    """

    @parameter
    fn _warp_sum[
        dtype: DType, width: Int
    ](x: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
        return lane_group_sum[num_lanes=WARP_SIZE](x)

    return block_reduce[block_size, _warp_sum, broadcast=broadcast](
        val, initial_val=0
    )


@always_inline
fn block_prefix_sum[
    type: DType, //,
    *,
    block_size: Int,
    exclusive: Bool = False,
](val: Scalar[type]) -> Scalar[type]:
    """Performs a prefix sum (scan) operation across all threads in a block.
    This function implements a block-level inclusive or exclusive scan,
    efficiently computing the cumulative sum for each thread based on
    thread indices.
    Parameters:
        type: The data type of the Scalar elements.
        block_size: The total number of threads in the block.
        exclusive: If True, perform exclusive scan instead of inclusive.
    Args:
        val: The Scalar value from each thread to include in the scan.
    Returns:
        A Scalar value containing the result of the scan operation for each
        thread.
    """
    constrained[
        block_size % WARP_SIZE == 0,
        "Block size must be a multiple of warp size",
    ]()

    # Allocate shared memory for inter-warp communication
    # We need one slot per warp to store warp-level scan results
    alias n_warps = block_size // WARP_SIZE
    var warp_mem = stack_allocation[
        n_warps, type, address_space = AddressSpace.SHARED
    ]()

    var thread_result = warp_prefix_sum[exclusive=exclusive](val)

    # Step 2: Store last value from each warp to shared memory
    var wid = warp_id()
    if lane_id() == WARP_SIZE - 1:
        var inclusive_warp_sum: Scalar[type] = thread_result

        @parameter
        if exclusive:
            # For exclusive scan, thread_result is the sum of elements 0 to
            # WARP_SIZE-2. 'val' is the value of the element at WARP_SIZE-1.
            # Adding them gives the inclusive sum of the warp.
            inclusive_warp_sum += val

        warp_mem[wid] = inclusive_warp_sum

    barrier()

    # Step 3: Have the first warp perform a scan on the warp results
    var lid = lane_id()
    if wid == 0:
        previous_warps_prefix = warp_prefix_sum[exclusive=False](warp_mem[lid])
        if lid < n_warps:
            warp_mem[lid] = previous_warps_prefix

    barrier()

    # Step 4: Add the prefix from previous warps
    if wid > 0:
        var warp_prefix = warp_mem[wid - 1]
        thread_result += warp_prefix

    return thread_result


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
