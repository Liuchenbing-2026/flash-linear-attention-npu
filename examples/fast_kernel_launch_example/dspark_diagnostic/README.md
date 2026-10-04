# DSpark eight-draft diagnostic dependency

This is an A2 diagnostic integration, separate from the vLLM-Ascend D2H patch. Eight drafts require nine verification tokens; the pinned official FLA kernel supports eight. The branch increases capacity to sixteen and makes example workspace allocation graph-safe. No public ABI is changed.

Build from the repository at `/source/fla-diagnostic`, in the validated CANN 9.1 / Python 3.12 / torch 2.10 environment:

```bash
cd /source/fla-diagnostic/examples/fast_kernel_launch_example
FAST_KERNEL_OP_NAME=recurrent_gated_delta_rule python setup.py bdist_wheel
python -m pip install --no-deps dist/ascend_ops-*.whl
python -m pip install --no-build-isolation --no-deps ./dspark_diagnostic
mkdir -p /work/results
ASCEND_RT_VISIBLE_DEVICES=0 python dspark_diagnostic/check_gdn_patched_matrix.py
ASCEND_RT_VISIBLE_DEVICES=0 python dspark_diagnostic/check_gdn_graph.py
```

The archived A2 run passed 40 numerical cases and 8 graph replay cases. Tests compare output and full state against the upstream CPU reference, including lengths 1/7/8/9/16, strides and state dtypes. A3/A5, complete ATK, and full official packaging acceptance are **not validated**. Do not describe this diagnostic branch as production-certified FLA.

`diag_gdn.py` registers the example kernel via `vllm.general_plugins`; the optional one-shot attention observer restores the original method before timed requests. Uninstall `dspark-gdn-length-diagnostic` and `ascend-ops` to validate official FLA with seven drafts.
