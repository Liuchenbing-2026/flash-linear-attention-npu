"""Experimental shared dependency shim; not part of the D2H PR."""
def register():
    import torch
    import ascend_ops
    import vllm_ascend.ops.gdn as gdn

    def recurrent(query, key, value, state, **kwargs):
        # The example kernel consumes dense read-only inputs; state has stride support.
        dense_kwargs = {key: value.contiguous() if isinstance(value, torch.Tensor) else value
                        for key, value in kwargs.items()}
        return torch.ops.ascend_ops.recurrent_gated_delta_rule(
            query.contiguous(), key.contiguous(), value.contiguous(), state, **dense_kwargs)

    gdn.recurrent_gated_delta_rule = recurrent
    print('DIAGNOSTIC_GDN max_verification_tokens=16 fast_kernel_launch; not stock FLA', flush=True)

    # One-shot activation evidence during output checks; restore before timed work.
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
    impl = AscendAttentionBackendImpl
    method_name = '_forward_draft_tail_masked'
    if hasattr(impl, method_name):
        original = getattr(impl, method_name)
        def observe_once(self, *args, **kwargs):
            output = original(self, *args, **kwargs)
            if output is not None:
                print('D2H_TAIL_MASK_ACTIVE: BSND device mask returned output; one-shot observer removed', flush=True)
                setattr(impl, method_name, original)
            else:
                from vllm_ascend.ascend_config import get_ascend_config
                if not get_ascend_config().enable_dspark_draft_kv_optimistic_bound:
                    setattr(impl, method_name, original)
            return output
        setattr(impl, method_name, observe_once)
