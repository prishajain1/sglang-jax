# Adapted from https://github.com/vllm-project/tpu-inference/releases/tag/v0.11.1
# Copyright 2025 The tpu-inference Authors. All rights reserved.
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.utils import cdiv


def get_slot_mapping(
    num_slices_per_block: int,
    kv_cache_start_loc: jax.Array,
    new_kv_start_loc: jax.Array,
    slice_lens: jax.Array,
):
    # Stack directly in [3, num_slices] layout to avoid a 2D TensorCore transpose.
    slot_mapping = jnp.stack(
        [kv_cache_start_loc, new_kv_start_loc, slice_lens], axis=0
    ).astype(jnp.int32)
    num_slices = slot_mapping.shape[1]
    padded_size = (
        (num_slices + num_slices_per_block - 1)
        // num_slices_per_block
        * num_slices_per_block
    )
    if padded_size > num_slices:
        slot_mapping = jnp.pad(
            slot_mapping,
            [[0, 0], [0, padded_size - num_slices]],
            constant_values=0,
        )
    return slot_mapping


VMEM_SIZE = 64 * 1024 * 1024  # 64MB


def get_num_slices_per_block(new_kv: jax.Array, kv_cache: jax.Array, page_size=128):
    """
    new_kv: 5D [tokens, 1, heads*2//packing, packing, head_dim]
    kv_cache: 5D [num_pages, page_size, heads*2//packing, packing, head_dim]
    """
    assert new_kv.ndim == 5, f"new_kv must be 5D, got {new_kv.ndim}D"
    assert kv_cache.ndim == 5, f"kv_cache must be 5D, got {kv_cache.ndim}D"
    assert (
        new_kv.dtype == kv_cache.dtype
    ), f"new_kv.dtype={new_kv.dtype} is not equal to kv_cache.dtype={kv_cache.dtype}"
    assert new_kv.dtype != jnp.float16, f"new_kv.dtype={new_kv.dtype} is not supported"

    bits = jnp.dtype(kv_cache.dtype).itemsize * 8
    assert bits % 8 == 0, f"bits={bits} is not divisible by 8"

    bytes_per_element = bits // 8

    total_num_token = new_kv.shape[0] * new_kv.shape[1]
    kv_head_num = new_kv.shape[2] * new_kv.shape[3]
    head_dim = new_kv.shape[4]

    max_num_slices_per_block = VMEM_SIZE // (bytes_per_element * page_size * kv_head_num * head_dim)
    assert (
        max_num_slices_per_block > 0
    ), f"max_num_slices_per_block={max_num_slices_per_block} is not greater than 0"

    return (
        total_num_token if total_num_token < max_num_slices_per_block else max_num_slices_per_block
    )


def kv_cache_update_kernel(
    # Prefetch
    # [3, padded_num_slices], list of (kv_cache_start, new_kv_start, slice_len)
    slices_ref,
    # Input
    new_kv_hbm_ref,  # [num_tokens, num_combined_kv_heads, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages * page_size, num_combined_kv_heads,
    # head_dim]
    # Output
    _,  # [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    # Scratch
    scratch,  # [num_slices_per_block, page_size, num_combined_kv_heads,
    # head_dim]
    sem,
):
    block_idx = pl.program_id(0)
    num_slices_per_block = scratch.shape[0]
    total_cache_slots = kv_cache_hbm_ref.shape[0]

    # Phase 1: Copy valid slices from new_kv_hbm_ref to scratch VMEM
    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        kv_cache_start = slices_ref[0, offset_i]
        new_kv_start = slices_ref[1, offset_i]
        length = slices_ref[2, offset_i]
        valid = (length > 0) & (kv_cache_start >= 0) & (kv_cache_start + length <= total_cache_slots)

        @pl.when(valid)
        def _start_read(i=i, new_kv_start=new_kv_start, length=length):
            pltpu.make_async_copy(
                new_kv_hbm_ref.at[pl.ds(new_kv_start, length), ...],
                scratch.at[jnp.uint32(i), pl.ds(0, length), ...],
                sem,
            ).start()

    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        kv_cache_start = slices_ref[0, offset_i]
        new_kv_start = slices_ref[1, offset_i]
        length = slices_ref[2, offset_i]
        valid = (length > 0) & (kv_cache_start >= 0) & (kv_cache_start + length <= total_cache_slots)

        @pl.when(valid)
        def _wait_read(i=i, new_kv_start=new_kv_start, length=length):
            pltpu.make_async_copy(
                new_kv_hbm_ref.at[pl.ds(new_kv_start, length), ...],
                scratch.at[jnp.uint32(i), pl.ds(0, length), ...],
                sem,
            ).wait()

    # Phase 2: Scatter valid slices from scratch VMEM to kv_cache_hbm_ref
    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        kv_cache_start = slices_ref[0, offset_i]
        length = slices_ref[2, offset_i]
        valid = (length > 0) & (kv_cache_start >= 0) & (kv_cache_start + length <= total_cache_slots)

        @pl.when(valid)
        def _start_write(i=i, kv_cache_start=kv_cache_start, length=length):
            pltpu.make_async_copy(
                scratch.at[jnp.uint32(i), pl.ds(0, length), ...],
                kv_cache_hbm_ref.at[pl.ds(kv_cache_start, length), ...],
                sem,
            ).start()

    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        kv_cache_start = slices_ref[0, offset_i]
        length = slices_ref[2, offset_i]
        valid = (length > 0) & (kv_cache_start >= 0) & (kv_cache_start + length <= total_cache_slots)

        @pl.when(valid)
        def _wait_write(i=i, kv_cache_start=kv_cache_start, length=length):
            pltpu.make_async_copy(
                scratch.at[jnp.uint32(i), pl.ds(0, length), ...],
                kv_cache_hbm_ref.at[pl.ds(kv_cache_start, length), ...],
                sem,
            ).wait()


def _kv_cache_update_slots_kernel(
    slots_ref,  # [padded_num_tokens] in SMEM
    new_kv_block_ref,  # [num_slices_per_block, num_combined_kv_heads, head_dim] in VMEM
    kv_cache_hbm_ref,  # [total_slots, num_combined_kv_heads, head_dim] in HBM
    _,  # aliased output in HBM
    sem,
):
    """Fast-path 1D-slot kernel: contiguous BlockSpec read of new_kv + predicated async DMA scatter."""
    block_idx = pl.program_id(0)
    num_slices_per_block = new_kv_block_ref.shape[0]
    total_cache_slots = kv_cache_hbm_ref.shape[0]

    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        slot = slots_ref[offset_i]
        valid = (slot >= 0) & (slot < total_cache_slots)

        @pl.when(valid)
        def _start_write(i=i, slot=slot):
            pltpu.make_async_copy(
                new_kv_block_ref.at[pl.ds(i, 1), ...],
                kv_cache_hbm_ref.at[pl.ds(slot, 1), ...],
                sem,
            ).start()

    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        slot = slots_ref[offset_i]
        valid = (slot >= 0) & (slot < total_cache_slots)

        @pl.when(valid)
        def _wait_write(i=i, slot=slot):
            pltpu.make_async_copy(
                new_kv_block_ref.at[pl.ds(i, 1), ...],
                kv_cache_hbm_ref.at[pl.ds(slot, 1), ...],
                sem,
            ).wait()


def kv_cache_update_slots(
    new_kv: jax.Array,
    slots: jax.Array,
    kv_cache: jax.Array,
    num_slices_per_block: int = 64,
) -> jax.Array:
    """Update paged KV cache from 1D token `slots` without `get_slot_mapping` or cache padding.

    - For small/medium slot widths (`slot_bytes < 4096` or `cache_bytes <= 64 MiB`), dispatches
      directly to XLA's `at[destination].set(..., mode='drop')` to avoid per-slot DMA descriptor
      overhead and TensorCore metadata preprocessing.
    - For large slot widths (`slot_bytes >= 4096` and `cache_bytes > 64 MiB`), uses a Pallas
      kernel that loads contiguous `new_kv` blocks via `BlockSpec` and issues coalesced async
      DMA writes (`num_slices_per_block=64`) directly into `kv_cache` in-place.
    """
    original_cache_shape = kv_cache.shape
    total_slots = kv_cache.shape[0] * kv_cache.shape[1]
    l = slots.shape[0]
    row_elems = kv_cache.size // total_slots
    bytes_per_elem = jnp.dtype(kv_cache.dtype).itemsize
    slot_bytes = row_elems * bytes_per_elem
    total_cache_bytes = total_slots * slot_bytes

    if slot_bytes < 4096 or total_cache_bytes <= 64 * 1024 * 1024 or row_elems % 256 != 0:
        flat_cache = kv_cache.reshape(total_slots, -1)
        flat_update = new_kv.reshape(l, -1)
        destination = jnp.where((slots >= 0) & (slots < total_slots), slots, total_slots)
        updated = flat_cache.at[destination].set(flat_update, mode="drop")
        return updated.reshape(original_cache_shape)

    num_combined_kv_heads = row_elems // 128
    head_dim = 128
    flat_cache = kv_cache.reshape(total_slots, num_combined_kv_heads, head_dim)
    flat_new_kv = new_kv.reshape(l, num_combined_kv_heads, head_dim)

    ns = min(num_slices_per_block, l)
    padded_l = cdiv(l, ns) * ns
    if padded_l > l:
        slots_padded = jnp.pad(slots, (0, padded_l - l), constant_values=-1).astype(jnp.int32)
        flat_new_kv = jnp.pad(flat_new_kv, ((0, padded_l - l), (0, 0), (0, 0)))
    else:
        slots_padded = slots.astype(jnp.int32)

    _any_mem = getattr(pltpu.MemorySpace, "ANY", pltpu.MemorySpace.HBM)
    in_specs = [
        pl.BlockSpec(
            (ns, num_combined_kv_heads, head_dim),
            lambda i, *_: (i, 0, 0),
        ),
        pl.BlockSpec(memory_space=_any_mem),
    ]
    out_specs = [pl.BlockSpec(memory_space=_any_mem)]
    out_shape = [jax.ShapeDtypeStruct(flat_cache.shape, dtype=flat_cache.dtype)]

    kernel = pl.pallas_call(
        _kv_cache_update_slots_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(padded_l // ns,),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_SIZE),
        out_shape=out_shape,
        input_output_aliases={2: 0},
    )
    result = kernel(slots_padded, flat_new_kv, flat_cache)[0]
    return result.reshape(original_cache_shape)


def kv_cache_update_impl(
    new_kv,
    slices,
    kv_cache,
    num_kv_update_slices,
    page_size,
    num_slices_per_block,
):
    """Accept 5D inputs. Flattens to 3D internally for Pallas kernel, reshapes output back to 5D."""
    if slices.ndim == 1 and page_size == 1:
        return kv_cache_update_slots(
            new_kv,
            slices,
            kv_cache,
            num_slices_per_block=num_slices_per_block,
        )

    assert new_kv.ndim == 5, f"new_kv must be 5D, got {new_kv.ndim}D: {new_kv.shape}"
    assert kv_cache.ndim == 5, f"kv_cache must be 5D, got {kv_cache.ndim}D: {kv_cache.shape}"
    assert (
        slices.shape[1] % num_slices_per_block == 0
    ), f"slices.shape[1]={slices.shape[1]} is not divisible by num_slices_per_block={num_slices_per_block}"

    original_cache_shape = kv_cache.shape
    s = new_kv.shape
    new_kv = new_kv.reshape(s[0] * s[1], s[2] * s[3], s[4])
    s = kv_cache.shape
    kv_cache = kv_cache.reshape(s[0] * s[1], s[2] * s[3], s[4])

    _, num_combined_kv_heads, head_dim = new_kv.shape

    assert num_combined_kv_heads % 2 == 0, (
        f"kv_cache_update_impl: num_combined_kv_heads={num_combined_kv_heads} should be even after pre-padding. "
        "This indicates a configuration issue with kv heads padding."
    )

    assert (
        kv_cache.shape[1] == num_combined_kv_heads
    ), f"kv_cache.shape[1]={kv_cache.shape[1]} is not equal to num_combined_kv_heads={num_combined_kv_heads}"
    assert (
        kv_cache.shape[2] == head_dim
    ), f"kv_cache.shape[2]={kv_cache.shape[2]} is not equal to head_dim={head_dim}"
    assert head_dim % 128 == 0, f"head_dim={head_dim} is not divisible by 128"

    _any_mem = getattr(pltpu.MemorySpace, "ANY", pltpu.MemorySpace.HBM)
    in_specs = [
        pl.BlockSpec(memory_space=_any_mem),
        pl.BlockSpec(memory_space=_any_mem),
    ]

    out_specs = [pl.BlockSpec(memory_space=_any_mem)]
    out_shape = [jax.ShapeDtypeStruct(kv_cache.shape, dtype=kv_cache.dtype)]

    scalar_prefetches = [slices]
    scratch = pltpu.VMEM(
        (num_slices_per_block, page_size, num_combined_kv_heads, head_dim),
        new_kv.dtype,
    )

    scratch_shapes = [
        scratch,
        pltpu.SemaphoreType.DMA,
    ]

    num_slices_int = (
        int(num_kv_update_slices[0])
        if not isinstance(num_kv_update_slices, int)
        and not isinstance(num_kv_update_slices[0], jax.Tracer)
        else (
            num_kv_update_slices
            if isinstance(num_kv_update_slices, int)
            else slices.shape[1]
        )
    )

    kernel = pl.pallas_call(
        kv_cache_update_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=out_specs,
            grid=(cdiv(num_slices_int, num_slices_per_block),),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=VMEM_SIZE,
        ),
        out_shape=out_shape,
        input_output_aliases={len(scalar_prefetches) + 1: 0},
    )

    result = kernel(*scalar_prefetches, new_kv, kv_cache)[0]

    return result.reshape(original_cache_shape)


@partial(
    jax.jit,
    static_argnames=["page_size", "num_slices_per_block", "kv_partition_axis"],
)
def kv_cache_update(
    new_kv: jax.Array,  # [total_num_token, num_kv_heads, head_dim]
    # [3, slices], list of (kv_cache_start, new_kv_start, slice_len)
    slices: jax.Array,
    # [max_num_tokens, num_kv_heads, head_dim]
    kv_cache: jax.Array,
    num_kv_update_slices: jax.Array,  # [1]
    *,
    page_size: int = 1,  # because we treat each token as an independent query
    num_slices_per_block: int = 8,
    kv_partition_axis: str = "tensor",
):
    @jax.shard_map(
        in_specs=(
            # new_kv - consistent with KV cache sharding
            P(None, kv_partition_axis, None),
            P(None, None),  # slices
            # kv_cache - consistent with KV cache sharding
            P(None, kv_partition_axis, None),
            P(None),  # num_kv_update_slices
        ),
        out_specs=P(
            None, kv_partition_axis, None
        ),  # output also maintains KV cache sharding consistency
        check_vma=False,
    )
    def _kv_cache_update_wrapper(new_kv, slices, kv_cache, num_kv_update_slices):
        assert (
            slices.shape[1] % num_slices_per_block == 0
        ), f"slices.shape[1]={slices.shape[1]} is not divisible by num_slices_per_block={num_slices_per_block}"
        _, num_combined_kv_heads, head_dim = new_kv.shape

        assert num_combined_kv_heads % 2 == 0, (
            f"num_combined_kv_heads={num_combined_kv_heads} should be even after pre-padding. "
            "This indicates a configuration issue with kv heads padding."
        )

        assert (
            kv_cache.shape[1] == num_combined_kv_heads
        ), f"kv_cache.shape[1]={kv_cache.shape[1]} is not equal to num_combined_kv_heads={num_combined_kv_heads}"
        assert (
            kv_cache.shape[2] == head_dim
        ), f"kv_cache.shape[2]={kv_cache.shape[2]} is not equal to head_dim={head_dim}"
        assert head_dim % 128 == 0, f"head_dim={head_dim} is not divisible by 128"
        # smaller or equal to page_size

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        ]

        out_specs = [pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY)]
        out_shape = [jax.ShapeDtypeStruct(kv_cache.shape, dtype=kv_cache.dtype)]

        scalar_prefetches = [slices]
        scratch = pltpu.VMEM(
            (num_slices_per_block, page_size, num_combined_kv_heads, head_dim),
            new_kv.dtype,
        )

        scratch_shapes = [
            scratch,
            pltpu.SemaphoreType.DMA,
        ]

        kernel = pl.pallas_call(
            kv_cache_update_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(cdiv(num_kv_update_slices[0], num_slices_per_block),),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=VMEM_SIZE,
            ),
            out_shape=out_shape,
            input_output_aliases={len(scalar_prefetches) + 1: 0},
        )

        result = kernel(*scalar_prefetches, new_kv, kv_cache)[0]

        return result

    return _kv_cache_update_wrapper(new_kv, slices, kv_cache, num_kv_update_slices)
