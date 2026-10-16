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

## 运维：指标与鉴权

### `/metrics`（Prometheus 指标）

默认关闭，与 llama-server 的 `--metrics` 语义一致。开启后暴露 Prometheus 兼容的
OpenMetrics 端点 `GET /metrics`：

```bash
./target/release/log-pi -m model.gguf --metrics
# 等价于环境变量 LOGPI_METRICS=1
```

当前导出的指标：

| 指标 | 类型 | 含义 |
|---|---|---|
| `logpi_requests_total{outcome}` | counter | 已受理的 SystemOne 请求，按结果分桶：`ok` / `overloaded`(503) / `timeout`(504) / `unavailable`(503) |
| `logpi_requests_in_flight` | gauge | 正在推理（排队中 + 执行中）的请求数 |
| `logpi_engine_queue_depth` | gauge | 引擎队列当前积压任务数 |
| `logpi_queue_wait_seconds` | histogram | 任务在队列中等待被引擎拾取的耗时 |
| `logpi_infer_seconds` | histogram | 单批问题的纯推理耗时 |

`/metrics` 与 `/health` 始终开放，方便编排器与 Prometheus 无鉴权探活/抓取。

### `--api-key`（Bearer Token 鉴权）

默认关闭。配置后，`POST /v1/systemone` 与 `GET /v1/models` 要求
`Authorization: Bearer <key>` 请求头：

```bash
./target/release/log-pi -m model.gguf --api-key secret123
# 多个 key 用逗号分隔；或使用环境变量 LOGPI_API_KEY
./target/release/log-pi -m model.gguf --api-key key1,key2
```

请求示例：

```bash
curl -H "Authorization: Bearer secret123" \
     -X POST http://localhost:8080/v1/systemone \
     -H "Content-Type: application/json" \
     -d '{"state":"...","questions":{...}}'
```

未提供或非法 key 返回 `401`。`/health`、`/` 与 `/metrics` 不受鉴权保护。

## 贡献

接受 PR。保持代码整洁、逻辑清晰、通过测试即可。

## 许可证

[MIT](LICENSE)
