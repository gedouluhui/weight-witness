# Weight Witness：模型权重工件审计

[在线演示](https://gedouluhui.github.io/weight-witness/demo/) · [GitHub](https://github.com/gedouluhui/weight-witness) · [CI结果](https://github.com/gedouluhui/weight-witness/actions/workflows/ci.yml)

不加载模型、不依赖 PyTorch、不执行 pickle、不上传数据。检查本地 safetensors 的分片索引是否真的指向正确权重，并逐张量比较两个 checkpoint 的内容；重分片但权重相同不会被误报成模型变化。

这是实验性 v0.1.0 工具，不是模型质量评测器、来源认证器或安全保证。英文 [README](README.md) 是完整使用与边界说明。

## 快速运行

```sh
git clone https://github.com/gedouluhui/weight-witness.git
cd weight-witness
python -m pip install .
python examples/make_demo.py --output artifacts/demo
python -m weight_witness diff artifacts/demo/base artifacts/demo/resharded
python -m weight_witness diff artifacts/demo/base artifacts/demo/candidate --json artifacts/diff.json --html artifacts/diff.html
```

第一条比较应相等，退出码 0；第二条包含真实合成权重变化，退出码 **1 是预期结果**。直接用浏览器打开 `artifacts/diff.html` 可搜索、筛选、展开每个张量的证据。示例全部为自制合成数据，未训练真实模型。重新生成时请选择新输出目录，脚本不覆盖既有目录。

`python -m weight_witness inspect artifacts/demo/broken` 演示各分片单独合法、但索引归属错误：退出码 2，不能冒充检查通过。

## 关键边界

- Python 3.10+，运行时仅标准库；尚未发布到 PyPI，不要按同名包盲装。
- F32/F16/BF16 支持数值扫描；F64、整数、支持的 FP8 等仅比较原始字节，不证明数值健康。未知或 packed dtype 明确拒绝。
- 容差左边为基准：`abs(right-left) <= atol + rtol*abs(left)`。NaN/Inf 无论字节是否相同均不通过数值检查。
- `equal` 是权重载荷指纹一致；`close` 是数值在声明容差内，二者不混为一谈。dtype/shape 变化单列。
- 默认每头部/索引 16MiB，最多10万个张量、4096片、64维；默认1MiB分块。元数据和报告仍占内存，不能宣传无限规模或常量总内存。
- 只核验显式索引引用的分片，不扫描整个模型目录。输入必须稳定；mtime等检查不能防御恶意并发修改。
- 报告不带绝对路径和元数据值，但张量名/分片名也可能敏感；页面筛选不等于删除数据，分享前自行检查。
- 不声称权重相同就意味着训练正确、LoRA合并正确、模型质量相同或来源可信。

## 验证

```sh
python -m unittest discover -s tests -v
python -m pip install numpy safetensors
python -m unittest discover -s tests -p test_oracle.py -v
```

第二组使用官方 safetensors 读写和 NumPy 数值作为独立参照。依赖缺失或 Windows 不允许创建 symlink 时，相关测试明确 skip，不能算作通过。发布时实测与未测项目见 [验证记录](docs/VERIFICATION.md)。

MIT 许可。欢迎提交最小合成复现、测试和可解释修复；不要上传私有权重、凭据或未经授权的数据。项目使用 AI 辅助开发并经过独立代码审查，不夸大原创算法或外部用户数量。
