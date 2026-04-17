# Artesia Sequence Analysis

## 原始输入

1. 运行 `iflow cli` 的 `main` 分支，包含 `3` 个输入 message 和 `1` 个输出 message。
2. 运行 `iflow cli` 的 `main` 分支，包含 `5` 个输入 message 和 `1` 个输出 message，其中输入中的 `message[2]` 和上一轮不一样。
3. 运行 `iflow cli` 的非 `main` 分支，包含 `3` 个输入 message 和 `1` 个输出 message。
4. 运行 `iclow cli` 的 `main` 分支，包含 `7` 个输入 message 和 `1` 个输出 message，其中输入中的 `message[4]` 和上一轮不同。
5. 运行非 `iflow` 的任务，包含 `1` 个输入 message 和 `1` 个输出 message，任务类型为“内存压缩任务”。
6. 运行 `iflow cli` 的 `main` 分支，包含 `2` 个输入 message 和 `1` 个输出 message，其中 `message[1]` 和上一轮不同。
7. 运行非 `iflow` 任务，包含 `1` 个输入 message 和 `1` 个输出 message。
8. 运行 `iflow` 的 `main` 分支，包含 `4` 个输入 message 和 `1` 个输出 message，其中 `message[1]` 和上一轮不同。
9. 运行 `iflow cli` 的非 `main` 分支，包含 `3` 个输入 message 和 `1` 个输出 message，且和上一次非 `main` 分支使用相同的 `semantic_type`。
10. 运行 `iflow` 的 `main` 分支，包含 `6` 个输入 message 和 `1` 个输出 message。
    - 输入 `message[0:5]` 与第 `8` 轮 LLM 调用的前 `5` 个输入 message 相同。
    - `message[5]` 为新增输入。

你要求分析中要明确区分：

- 哪些 Artesia 原语在 `LLM 前` 触发
- 哪些行为在 `LLM 执行` 时发生
- 哪些 Artesia 原语在 `LLM 后` 触发

## 结构化输入

### 规范化假设

- 将你输入里的 `iflow cli`、`iflow`、`iclow cli` 都视为同一个 `iFlow CLI` 任务族；否则，按代码的精确字符串匹配，有些轮次会直接被归类为 `non_loop`。
- 为避免原始编号里出现两个 `4.`，这里统一重排为 `R1` 到 `R10`。
- `R3` 和 `R9` 是同一个非 main `semantic_type`，因此复用同一个派生 context。
- `R10` 的含义现在明确为：
  - 前 `5` 条输入 message 与规范化后的 `R8` 所对应的长度为 `5` 的历史前缀一致。
  - 第 `6` 条输入 message 为新增内容。
  - 因此 `R10` 采用单一路径分析：`p = 5`。
- 下文里的 `message[2]`、`message[4]`、`message[1]` 都按 0-based 下标理解。

### 结构化场景表

| 规范轮次 | 原始条目 | 任务族 | 分支 | semantic_type | 输入数 | 输出数 | 与前一轮同 context 的差异 | prefix 锚点 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `R1` | `1` | `iFlow CLI` | `main` | `null` | 3 | 1 | 首轮 | 无 |
| `R2` | `2` | `iFlow CLI` | `main` | `null` | 5 | 1 | `message[2]` 起变化 | 无 |
| `R3` | `3` | `iFlow CLI` | `derived` | `branch-A` | 3 | 1 | 首次进入该派生分支 | 无 |
| `R4` | `4` | `iFlow CLI` | `main` | `null` | 7 | 1 | `message[4]` 起变化 | 无 |
| `R5` | `4(第二个)` | `non-iFlow` | `n/a` | `null` | 1 | 1 | 非循环任务 | 无 |
| `R6` | `5` | `iFlow CLI` | `main` | `null` | 2 | 1 | `message[1]` 起变化 | 无 |
| `R7` | `6` | `non-iFlow` | `n/a` | `null` | 1 | 1 | 非循环任务 | 无 |
| `R8` | `7` | `iFlow CLI` | `main` | `null` | 4 | 1 | `message[1]` 起变化 | 无 |
| `R9` | `8` | `iFlow CLI` | `derived` | `branch-A` | 3 | 1 | 与 `R3` 相同 `semantic_type` | 复用 `R3` 的分支上下文 |
| `R10` | `9` | `iFlow CLI` | `main` | `null` | 6 | 1 | `message[0:5]` 命中规范化 `R8` 对应的长度为 `5` 的历史前缀，`message[5]` 为新增输入 | `prefix_pos -> R8-derived 5-message prefix anchor` |

## 代码行为总览

### Artesia 原语插入位置

- `LLM 前`：由 `append_artesia_commands_for_round(...)` 插入，在 `openai.chat.completions.create(...)` 之前执行。
- `LLM 后`：由 `append_artesia_post_commands_for_round(...)` 插入，在 assistant 输出已经拿到之后执行。

对应代码位置：

- 分类与 context 预计算：`dev/scripts/load_gen/server/main_art.py:161-220`
- 前置原语：`dev/scripts/load_gen/server/main_art.py:246-319`
- 后置原语：`dev/scripts/load_gen/server/main_art.py:322-356`
- 主循环中的接入点：`dev/scripts/load_gen/server/main_art.py:498-555`

### 这份代码里最重要的真实行为

1. Artesia 命令只是附加标注，不会改变原始 LLM 调用逻辑。
2. `non_loop` 任务虽然会在 `LLM 前` 打 `artesia.reject(call_id)`，但代码不会因此跳过 LLM，请求仍然会照常执行。
3. `loop_derived` 的后置行为固定是 `artesia.tag(context_id, 0)` 和 `artesia.truncate(context_id, stash_list)`；它不会走 `reject`，也不会在 `LLM 后` 做 `offload(duration=wait_time)`。
4. `loop_main` 每一轮在 `LLM 后` 都会先打一条 `artesia.offload(context_id, duration=wait_time)`；如果下一轮切去别的 context，下一轮 `LLM 前` 还会再打一条 `artesia.offload(current_context_id, duration=None)`。
5. 真正的 prompt token 复用逻辑仍然是按 `task_list[i]` 的整数值做的，不是按 Artesia 的 `context_id` 做的。

### 一个很关键的偏差

虽然 Artesia 把 `main` 和 `derived` 分成不同 `context_id`，但实际消息/token 复用还是由下面这段逻辑控制：

- `prev_msg_token_ids_per_round[task_type][-1]`
- `history_rounds[task_type]`

也就是说，如果 `main` 和 `derived` 轮次共用同一个 `task_type` 整数，那么：

- 代码构造下一轮 prompt 时，仍然会参考“该 `task_type` 最近一轮”的 token 结果；
- 这个最近一轮可能恰好是另一个分支；
- Artesia 的 `fetch/offload/create/delete` 并不会反向修正这个 prompt 构造逻辑。

所以，Artesia 命令表达的是“你希望怎样管理上下文”，而不是“底层 prompt 生成一定已经按这个上下文隔离”。

## 轮次分析

### 记号

- `C_main`：`iFlow main` 的 `context_id`
- `C_branch`：`iFlow derived(branch-A)` 的 `context_id`
- `C_mem`：内存压缩任务的 `context_id`
- `C_other`：另一个非 iFlow 任务的 `context_id`

### R1

场景：`iFlow main`，3 个输入，1 个输出。

`LLM 前`

- `artesia.create(agent_id, C_main)`
- `artesia.fetch(C_main)`
- `artesia.truncate(C_main, 0)`

说明：

- 这是 `C_main` 首次出现，因此先 `create` 再 `fetch`。
- 作为 `loop_main` 首轮，没有可复用前缀，`p = 0`。

`LLM 执行`

- 发送 3 个输入 message 给 `openai.chat.completions.create(...)`

`LLM 后`

- `artesia.tag(C_main, [0, 1, 2, 3])`
- `artesia.offload(C_main, duration=wait_1)`

### R2

场景：`iFlow main`，5 个输入，`message[2]` 首次不同。

`LLM 前`

- `artesia.fetch(C_main)`
- `artesia.truncate(C_main, 2)`

说明：

- 仍在同一个 `C_main` 中，因此不会 `create`，也不会先 offload 当前 context。
- 按你的描述，首个失配点在 `2`，因此 `p = 2`。

`LLM 执行`

- 发送 5 个输入 message 给 LLM

`LLM 后`

- `artesia.tag(C_main, [2, 3, 4, 5])`
- `artesia.offload(C_main, duration=wait_2)`

### R3

场景：`iFlow derived(branch-A)`，3 个输入，1 个输出。

`LLM 前`

- `artesia.offload(C_main, duration=None)`
- `artesia.create(agent_id, C_branch)`
- `artesia.fetch(C_branch)`

说明：

- 从 `main` 切到 `derived`，因此要先 offload 当前的 `C_main`。
- `C_branch` 首次出现，所以需要 `create`。

`LLM 执行`

- 发送 3 个输入 message 给 LLM

`LLM 后`

- `artesia.tag(C_branch, 0)`
- `artesia.truncate(C_branch, [1, 2, 3])`

说明：

- 这里不会 `delete(C_branch)`，因为 `R9` 还要再次使用同一个 `semantic_type`。

### R4

场景：回到 `iFlow main`，7 个输入，`message[4]` 首次不同。

`LLM 前`

- `artesia.offload(C_branch, duration=None)`
- `artesia.fetch(C_main)`
- `artesia.truncate(C_main, 4)`

`LLM 执行`

- 发送 7 个输入 message 给 LLM

`LLM 后`

- `artesia.tag(C_main, [4, 5, 6, 7])`
- `artesia.offload(C_main, duration=wait_4)`

### R5

场景：非 iFlow 任务，内存压缩，1 个输入，1 个输出。

`LLM 前`

- `artesia.offload(C_main, duration=None)`
- `artesia.create(agent_id, C_mem)`
- `artesia.fetch(C_mem)`
- `artesia.reject(call_4)`

说明：

- 这是 `non_loop` 任务，所以会在 `LLM 前` 打 `reject(call_id)`。
- 但代码不会因为 `reject` 而跳过 LLM。

`LLM 执行`

- 发送 1 个输入 message 给 LLM

`LLM 后`

- `artesia.delete(C_mem)`

### R6

场景：回到 `iFlow main`，2 个输入，`message[1]` 首次不同。

`LLM 前`

- `artesia.fetch(C_main)`
- `artesia.truncate(C_main, 1)`

说明：

- `R5` 结束时 `C_mem` 已删除，因此当前状态为空；回到 `C_main` 时不需要先 offload 旧 context。

`LLM 执行`

- 发送 2 个输入 message 给 LLM

`LLM 后`

- `artesia.tag(C_main, [1, 2])`
- `artesia.offload(C_main, duration=wait_6)`

### R7

场景：另一个非 iFlow 任务，1 个输入，1 个输出。

`LLM 前`

- `artesia.offload(C_main, duration=None)`
- `artesia.create(agent_id, C_other)`
- `artesia.fetch(C_other)`
- `artesia.reject(call_6)`

`LLM 执行`

- 发送 1 个输入 message 给 LLM

`LLM 后`

- `artesia.delete(C_other)`

### R8

场景：回到 `iFlow main`，4 个输入，`message[1]` 首次不同。

`LLM 前`

- `artesia.fetch(C_main)`
- `artesia.truncate(C_main, 1)`

`LLM 执行`

- 发送 4 个输入 message 给 LLM

`LLM 后`

- `artesia.tag(C_main, [1, 2, 3, 4])`
- `artesia.offload(C_main, duration=wait_8)`

### R9

场景：再次进入 `iFlow derived(branch-A)`，3 个输入，且与 `R3` 使用相同 `semantic_type`。

`LLM 前`

- `artesia.offload(C_main, duration=None)`
- `artesia.fetch(C_branch)`

说明：

- 因为这是和 `R3` 相同的 `semantic_type`，所以复用同一个 `C_branch`。
- 不会重新 `create(C_branch)`。

`LLM 执行`

- 发送 3 个输入 message 给 LLM

`LLM 后`

- `artesia.tag(C_branch, 0)`
- `artesia.truncate(C_branch, [1, 2, 3])`
- `artesia.delete(C_branch)`

说明：

- `R9` 是 `C_branch` 的最后一轮，所以这里会 `delete(C_branch)`。

### R10

场景：回到 `iFlow main`，6 个输入，1 个输出；其中 `message[0:5]` 与一个长度为 `5` 的历史前缀一致，`message[5]` 为新增输入。

因此，这里的唯一自洽解释是：

- `m = 6`
- `prefix_pos` 指向一个长度为 `5` 的 prefix anchor
- 本轮是 `5` 条前缀命中，加 `1` 条新增输入
- 因而首个新增位置为 `p = 5`

`LLM 前`

- `artesia.fetch(C_main)`
- 通常不会发 `artesia.truncate(C_main, 5)`

原因：

- 如果 `prefix_pos` 指到的那一轮满足 `len(a_list[prefix_pos]) = 5`
- 且本轮算出的 `p = 5`
- 那么会命中 skip 条件，前置 `truncate` 被跳过

`LLM 执行`

- 发送 6 个输入 message 给 LLM

`LLM 后`

- `artesia.tag(C_main, [5, 6])`
- `artesia.offload(C_main, duration=wait_10)`
- `artesia.delete(C_main)`

说明：

- `[5, 6]` 表示：第 `5` 条新增输入 message，以及 assistant 输出会在这一轮被 tag。
- 这和“`message[0:5]` 命中、`message[5]` 为新增输入”的描述完全一致。

## 结论

### 按你的结构化语义，整体命令节奏如下

- `main` 分支：
  - 首轮 `create + fetch + truncate`
  - 中间轮 `fetch + truncate`
  - 命中 prefix 特判时，`fetch` 后可跳过 `truncate`
  - 每轮 `LLM 后` 都会 `tag + offload(wait)`
  - 最后一轮额外 `delete`
- `derived` 分支：
  - 首轮 `offload(main) + create + fetch`
  - 再次进入同一 `semantic_type` 时只会 `offload(main) + fetch`
  - `LLM 后` 固定 `tag(0) + truncate(stash_list)`
  - 最后一轮额外 `delete`
- `non-iFlow` 任务：
  - `LLM 前` 会 `create/fetch/reject`
  - 但 LLM 仍然执行
  - `LLM 后` 如果是最后一轮则 `delete`

### 关于 R10 的最终结论

- `R10` 的唯一目标语义是：
  - 前 `5` 条输入命中旧前缀
  - 第 `6` 条输入为新增内容
- 因此 `R10` 的行为应为：
  - `LLM 前`：`fetch`
  - `LLM 后`：`tag([5, 6]) + offload + delete`
- 在这个前提下，`LLM 前` 的 `truncate(5)` 还可能因为 `prefix_pos` 命中 skip 条件而完全不发。

### 这份代码和你心智模型之间的潜在落差

- 代码对 `iFlow CLI` 的识别是精确字符串匹配，不会自动容错 `iflow`、`iclow` 这些写法。
- Artesia 命令虽然把 `main`/`derived` 拆成不同 context，但底层 prompt/token 复用仍然按 `task_type` 整数维度共享。
- 因此，如果你的 replay 数据里 `main` 和 `derived` 共用同一个 `task_list` 编号，那么真实 prompt 构造可能会受到另一个分支上一轮内容的影响。
