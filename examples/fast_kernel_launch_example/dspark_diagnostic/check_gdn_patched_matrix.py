import importlib.util
import json
from pathlib import Path
import torch
p=Path('/source/fla-diagnostic/examples/fast_kernel_launch_example/tests/recurrent_gated_delta_rule/test_recurrent_gated_delta_rule.py')
spec=importlib.util.spec_from_file_location('gdn_test',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=[]
for length in [1,7,8,9,16]:
 for layout in ['contiguous','noncontiguous']:
  for dtype in [torch.float32,torch.bfloat16]:
   for use_gk in [False,True]:
    inp=m.make_inputs(2,length,2,4,128,128,use_g=True,use_gk=use_gk,use_accepted_tokens=True,seed=42)
    inp['state']=inp['state'].to(dtype)
    golden=m.run_golden(inp)
    actual,state,*_=m.run_npu(inp,layout,'mutable')
    m.assert_compare_tensors_by_ratio(golden[0],actual,'output')
    m.assert_compare_tensors_by_ratio(golden[1],state,'state')
    row=dict(length=length,layout=layout,state_dtype=str(dtype),use_gk=use_gk,status='pass')
    print(row,flush=True);rows.append(row)
Path('/work/results/gdn_patched_matrix.json').write_text(json.dumps(rows,indent=2))
