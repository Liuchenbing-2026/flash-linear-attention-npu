import importlib.util
import json
from pathlib import Path
import torch
import torch_npu
import ascend_ops
p=Path('/source/fla-diagnostic/examples/fast_kernel_launch_example/tests/recurrent_gated_delta_rule/test_recurrent_gated_delta_rule.py')
spec=importlib.util.spec_from_file_location('gdn_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.npu.set_device(0)
rows=[]
for length in [9,16]:
 for layout in ['contiguous','noncontiguous']:
  inp=m.make_inputs(2,length,2,4,128,128,use_g=True,use_accepted_tokens=True,seed=42)
  q,k,v=[inp[x].npu() for x in ['query','key','value']]
  state,_=m.make_npu_state(inp['state'],'npu',layout)
  kw={x:inp[x].npu() if isinstance(inp[x],torch.Tensor) else inp[x] for x in ['beta','scale','actual_seq_lengths','ssm_state_indices','num_accepted_tokens','g','gk']}
  for _ in range(3):
   torch.ops.ascend_ops.recurrent_gated_delta_rule(q,k,v,state,**kw)
  torch.npu.synchronize()
  graph=torch.npu.NPUGraph()
  with torch.npu.graph(graph):
   output=torch.ops.ascend_ops.recurrent_gated_delta_rule(q,k,v,state,**kw)
  for factor in [1,2]:
   state.copy_(inp['state'].npu())
   q.copy_((inp['query']*factor).npu())
   graph.replay();torch.npu.synchronize()
   ref_inp=dict(inp);ref_inp['query']=inp['query']*factor
   ref=m.run_golden(ref_inp)
   actual=output.cpu();actual[:int(inp['actual_seq_lengths'][0])]=0
   m.assert_compare_tensors_by_ratio(ref[0],actual,'graph output')
   m.assert_compare_tensors_by_ratio(ref[1],state.cpu(),'graph state')
   row=dict(length=length,layout=layout,input_factor=factor,status='pass');rows.append(row);print(row,flush=True)
Path('/work/results/gdn_graph.json').write_text(json.dumps(rows,indent=2))
