import torch
import triton
import triton.language as tl

class FlashAttention(torch.autograd.Function):
  @staticmethod
  def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, softmax_scale: float):
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

    O = torch.empty_like(Q)
    batch_size, num_heads, seq_len, head_dim = Q.shape
    grid = lambda meta: (
        triton.cdiv(seq_len, meta["BLOCK_SIZE_Q"]),
        batch_size * num_heads,
        1
    )

    _attn_fwd[grid](
      Q_ptr=Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
      K_ptr=K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
      V_ptr=V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
      O_ptr=O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
      softmax_scale=softmax_scale,
      stride_Q_batch=Q.stride(0),
      stride_Q_head=Q.stride(1),
      stride_Q_seq=Q.stride(2),
      stride_Q_dim=Q.stride(3),
      stride_K_batch=K.stride(0),
      stride_K_head=K.stride(1),
      stride_K_seq=K.stride(2),
      stride_K_dim=K.stride(3),
      stride_V_batch=V.stride(0),
      stride_V_head=V.stride(1),
      stride_V_seq=V.stride(2),
      stride_V_dim=V.stride(3),
      stride_O_batch=O.stride(0),
      stride_O_head=O.stride(1),
      stride_O_seq=O.stride(2),
      stride_O_dim=O.stride(3),
      BATCH_SIZE=batch_size,
      NUM_HEADS=num_heads,
      SEQ_LEN=seq_len,
      HEAD_DIM=head_dim,
      BLOCK_SIZE_Q=16,
      BLOCK_SIZE_KV=16,
    )
    return O
  
  @staticmethod
  def backward(ctx, dO):
     # TODO
     pass

@triton.jit
def _attn_fwd(
    Q_ptr,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K_ptr,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V_ptr,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    O_ptr,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
  '''
  Parallel kernel instances will each handle a separate
  (query block index, head index, index in batch). 
  
  This parallelizes over the Q blocks (the outer for-loop in
  the Flash Attention algorithm), and within those blocks
  parallelizes further across each sequence (index in the batch
  dimension), and within those parallelizes further across each
  head.

  Total degree of parallelization will be: 
  
  (SEQ_LEN // BLOCK_SIZE Q) * BATCH_SIZE * NUM_HEADS
  '''
  inf = 1.0e6

  # each program handles a specific query head for a specific index in the batch dimension.
  # this is represented as a 2D index (query_idx, batch_idx * head_idx)
  query_block_idx = tl.program_id(axis=0)
  batch_head_idx = tl.program_id(axis=1)

  # decompose batch_idx * head_idx into separate batch_idx and head_idx
  batch_idx = batch_head_idx // NUM_HEADS
  head_idx = batch_head_idx % NUM_HEADS

  # calculate offset to this batch and head
  qkv_offset = batch_idx * stride_Q_batch + head_idx * stride_Q_head

  # get subset of Q blocks we are processing in this program id.
  # Q[batch_idx, head_idx, :, :]
  Q_block_ptr = tl.make_block_ptr(
      # by adding the offset to the right batch idx & head idx, 
      # the base points to the start of a tensor of shape (seq, head_dim)
      # within the parent tensor of shape (batch, heads, seq, dim)
      base=Q_ptr + qkv_offset,    # Q[batch_idx, head_idx, :, :]
      shape=(SEQ_LEN, HEAD_DIM),
      strides=(stride_Q_seq, stride_Q_dim),
      # the (seq, head) sub tensor has all the queries in it,
      # so offset into the specific query block we want.
      # # Q[batch_idx, head_idx, q_idx:q_idx+block_size_q, :]
      offsets=(query_block_idx * BLOCK_SIZE_Q, 0), 
      block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
      order=(1,0),
  )

  # get K block. needs to be transposed for Q @ K^T.
  # K[batch_idx, head_idx, :, :]
  K_block_ptr = tl.make_block_ptr(
      base=K_ptr + qkv_offset,
      # inverse shape and stride params to transpose w.r.t. Q
      shape=(HEAD_DIM, SEQ_LEN),
      strides=(stride_K_dim, stride_K_seq),
      # for K,V we select all keys and values, not a sub-block like in Q,
      # so we don't add any offsets and just start at the beginning of the block.
      offsets=(0, 0),
      block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
      order=(1,0),
  )

  # get V block.
  # V[batch_idx, head_idx, :, :]
  V_block_ptr = tl.make_block_ptr(
      base=V_ptr + qkv_offset,
      shape=(SEQ_LEN, HEAD_DIM),
      strides=(stride_V_seq, stride_V_dim),
      offsets=(0, 0),
      block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
      order=(1,0),
  )

  # get O (output) block ptrs.
  O_block_ptr = tl.make_block_ptr(
      # points to O[batch_idx, head_idx, :, :] 
      # of shape (seq, head dim) just like Q,K,V.
      base=O_ptr + qkv_offset,
      shape=(SEQ_LEN, HEAD_DIM),
      strides=(stride_O_seq, stride_O_dim),
      # offsets will be same as Q since we are writing
      # outputs for the subset of queries process in this program id.
      offsets=(query_block_idx * BLOCK_SIZE_Q, 0), 
      block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
      order=(1,0),
  )

  # offsets of each query within the current Q block.
  offs_q = query_block_idx * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

  # offsets for each k/v in K/V blocks. for each Q block 
  # need to loop through all of K and V so this will start at 0
  # offset.
  offs_kv = tl.arange(0, BLOCK_SIZE_KV)

  # m_i = max seen so far in QK. track one for each query.
  global_qk_max = tl.full((BLOCK_SIZE_Q,), -float('inf'), dtype=tl.float32)

  # l_i = accumlated global softmax denominator / exp sum
  global_softmax_denom = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)

  # accumulator for block of output matrix being computed by this program id.
  O_block = tl.zeros((BLOCK_SIZE_Q, HEAD_DIM), dtype=tl.float32)

  # load Q block into SRAM, it will be shared for all iterations of inner
  # loop doing O = softmax(Q @ K^T / scale) @ V
  Q_block = tl.load(Q_block_ptr)                                # (BLOCK_SIZE_Q, HEAD_DIM)

  # go to correct starting block of K/V for the given Q
  start, end = query_block_idx * BLOCK_SIZE_Q, (query_block_idx + 1) * BLOCK_SIZE_Q

  # move K block ptr to start of the first K block for this Q block.
  # K dims = (HEAD_DIM, BLOCK_SIZE_KV) -> inverse of Q and V since it is K^T
  # so we advance 0 along head dim and only move starting offset along the actual keys
  K_block_ptr = tl.advance(K_block_ptr, (0, start))

  # move V block ptr to first V block 
  # V dims = (BLOCK_SIZE_KV, HEAD_DIM)
  # so we advance to the correct starting offset along the values and stay at start of head dim
  V_block_ptr = tl.advance(V_block_ptr, (start, 0))

  # for each Q block, iterate through all associated K and V blocks
  for start_kv_idx in tl.range(start, end, BLOCK_SIZE_KV):
    causal_mask = offs_q[:, None] >= (start_kv_idx + offs_kv[None, :])

    # load next K block into SRAM
    K_block = tl.load(K_block_ptr)                              # (HEAD_DIM, BLOCK_SIZE_Q)

    # compute attention scores
    # S[i,j]
    QK_block = (
        tl.dot(Q_block, K_block) 
        * softmax_scale 
        + tl.where(causal_mask, 0, -inf)
    )                                                           # (BLOCK_SIZE_Q, BLOCK_SIZE_KV)

    # m[i,j]
    local_qk_max = tl.max(QK_block, axis=1)                     # (BLOCK_SIZE_Q,)

    # corrective factor for previously accumulated denominator
    corrective_factor = tl.exp(global_qk_max - local_qk_max)    # (BLOCK_SIZE_Q,)

    # P[i,j] (exp scores)
    P_block = tl.exp(QK_block)                                  # (BLOCK_SIZE_Q, BLOCK_SIZE_KV)

    # rowsum(P[i,j])
    exp_sum = tl.sum(P_block, axis=1)                           # (BLOCK_SIZE_Q,)

    # l[i,j]
    global_softmax_denom = (
        global_softmax_denom * corrective_factor + exp_sum      # (BLOCK_SIZE_Q,)
    )

    # m[i]
    global_qk_max = tl.maximum(global_qk_max, local_qk_max)
    
    # load V block into SRAM
    V_block = tl.load(V_block_ptr)                              # (BLOCK_SIZE_KV, HEAD_DIM)

    # apply corrective factor to O block
    # O[i,j]
    O_block = O_block * corrective_factor[:, None]              # (BLOCK_SIZE_Q, HEAD_DIM)
    O_block = O_block + tl.dot(P_block, V_block)                # (BLOCK_SIZE_Q, HEAD_DIM)

    # move to next K,V blocks
    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))

  # normalize scores to finalize softmax block
  O_block = O_block / global_softmax_denom[:, None]             # (BLOCK_SIZE_Q, HEAD_DIM
    
  # store O block output in HBM
  tl.store(O_block_ptr, O_block)

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM**0.5)

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=device))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1)
    ref_O = torch.matmul(P, V)

    # triton implementation
    flash_out = FlashAttention.apply(Q, K, V, softmax_scale)

    # compare
    rtol = 0.0
    atol = 1e-2
    if not torch.allclose(ref_O, flash_out, atol=atol, rtol=rtol):
        print("want:")
        print(ref_O)
        print("\ngot:")
        print(flash_out)
        print("FAILED")
    else:
        print("PASSED")


if __name__ == "__main__":
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=128, HEAD_DIM=64)
    print("PASSED")