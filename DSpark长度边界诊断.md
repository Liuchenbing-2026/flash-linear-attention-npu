# DSpark 验证长度边界诊断（实验分支，未作为可合入 FLA 修复交付）

基线 a4a795857868729fbdbb0118c1e06c8655a7b31b。A2、CANN 9.1、BF16 QKV / FP32 state。
当前公开上限为 8；DSpark 配置 8 个草稿后，目标验证长度为 9，超出支持范围。
原实现直接 return；实测长度 9/16 输出非有限值且 state 未更新，长度 1/7/8 正常。

本实验仅扩大共享 tiling UB 预算与 A2/A3、A5 kernel 的容量常量至 16，数学语义、索引、ABI 不变。
采用上游 fast-kernel example 单独编译，先运行 CPU 参考对照，随后做全模型因果验证。
暂不安装/覆盖完整官方 FLA OPP，不宣称 A3/A5 或完整 ATK 验收通过。
回归范围：1/7/8/9/16 长度，accepted index、FP32/BF16 state、padding stride、GQA。
该补丁与 D2H 分支分开，测量时必须明确依赖补丁及调用路径。

图封装修复：示例的 aclrtMalloc/free 与同步改为框架管理 workspace，移除捕获中的显式同步。只读非连续输入在独立插件中转连续，state 保留原 stride。40 组数值检查与 8 组捕获/replay 检查通过；A2 单平台诊断证据，不代表完整 FLA 验收。
