# log(π)

```
  ++                    ##              @@
  ==                  **    ########%%    %%
  ==    ==      ++++  **      ##  ##      %%
  --  ==  ==  ==  ++  ++      **  ****    ##
  --  --  ==  ==  ==  ++      **  ***     ##
  ::    --      ====    ==              **
                  --
              ::--
```

一个用 Rust 编写的 System One 决策模型服务。

## 项目背景

- 受到 [Jev](https://typesafe.ai) 以及文章 [《Jev in 25 lines of Python》](https://medium.com/data-science-in-your-pocket/jev-in-25-lines-of-python-dea40d30fea0) 的启发。
- 使用 llama.cpp 作为后端，参数也尽可能仿照 llama-server 风格。
- 纯粹是个人兴趣使然，外加每月订阅的 Token Plan 用不完顺手折腾出来的产物。

## 名字来源

`log(pi)` 这个名字具有双重含义：

1. **Logic Piano**：致敬 1869 年威廉·斯坦利·杰文斯（William Stanley Jevons，Jev 命名之源）发明的首台机械逻辑计算机「逻辑钢琴」（Logic Piano）。
2. **$\log(P_i)$**：机器学习与信息论中交叉熵损失（Cross-Entropy Loss $-\sum p_i \log q_i$）或 KL 散度（Kullback–Leibler Divergence）里的核心项 $\log(P_i)$。本项目不做自回归文本生成，而是在单次前向传播中直接读取候选 Token 的 Logits 并归一化计算对数概率。

## 快速构建和运行

```bash
# 1. 安装构建与运行时依赖 (Arch Linux)
sudo pacman -Sy --needed base-devel rust llama-cpp

# 2. 编译 Release 二进制文件
make release

# 3. 运行
./target/release/log-pi [OPTIONS] -m /path/to/model.gguf
```

## 贡献

接受 PR。保持代码整洁、逻辑清晰、通过测试即可。

## 许可证

[MIT](LICENSE)
