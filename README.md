# nearidx — 落盘词位索引与结构化近邻查询器

纯 Python + SQLite 实现。构建时把每篇文档切成词位并落盘倒排表，查询时先从倒排表取候选文档，再在候选文档内部求解 `SEQ` / `NEAR` / `OR` 的可嵌套查询树，最后只对真正要交付的窗口读取正文并切出原文片段。不依赖任何全文搜索引擎，不做分词模型、词干、同义词或词表训练。

## 快速开始

```bash
python3 -m nearidx build demo.nearidx docs/*.txt --overwrite
python3 -m nearidx search demo.nearidx 'NEAR(3,[pump,pump])' --limit 5
python3 -m nearidx snippet demo.nearidx d1 --positions 0:2
python3 -m nearidx verify demo.nearidx --full
python3 -m nearidx rebuild demo.nearidx d1 --file docs/d1.txt
```

Python API：

```python
from nearidx import build_index, open_index, parse_query, search

build_index("demo.nearidx", [("d1", "pump valve pump")], overwrite=True)
with open_index("demo.nearidx") as idx:
    page = search(idx, parse_query("NEAR(3,[pump,pump])"), limit=10)
    for w in page.windows:
        print(w.doc_id, w.start, w.end, w.witness, w.snippet, w.leaves)
```

运行测试与演示：

```bash
python3 -m unittest discover -s nearidx/tests -t .   # 47 个测试
python3 -m nearidx.factcheck                         # 主例与嵌套例的落盘演示
python3 examples/scale_demo.py 2000                   # 大量无关文档 + 访问证据
```

## 词法（构词规则）

- 一个 token 是连续 Unicode 字母/数字/下划线（`ch.isalnum() or ch == "_"`）的最大串；其余任何字符都是分隔符，连续分隔符不产生空 token。
- 仅对 token 整体做 `text.casefold()` 作为查询键；原文不改写，保留原文字符起止和 UTF-8 字节起止。
- 位置从 0 开始按出现顺序编号。字符/字节跨度用一次线性扫描累计，多字节文本不会退化为二次复杂度。
- 查询 token 必须经过同一词法得到**恰好一个** token（`nearidx.lexer.single_query_token`）。`"a-b"`、`"a b"` 等在解析期直接报 `QuerySyntaxError`，绝不被拆分或展开。

## 查询语言

```
TOKEN                     # 单个词，按 casefold 查
OR(expr, expr, ...)       # 至少两个子节点，恰好选择一个子命中
SEQ(gap, [expr, ...])     # 顺序；前一子区间结束 < 后一开始；两区间之间未占用词位数 <= gap
NEAR(width, [expr, ...])  # 无序；实际用到的位置集合两两不交；包围区间宽度 <= width
```

`SEQ` / `NEAR` 也接受圆括号形式 `SEQ(0, a, b)`；至少两个子节点。查询树可任意嵌套，叶节点按**查询树先序**编号（0 起）。

### 命中与见证

- 一次命中 = 一组实际使用的 `(叶编号, 词位)` 对 + 这些位置的闭包围区间。
- `OR` 只选择一个子命中；未选分支下的叶记为**未使用**，见证中写作 `-1`；真实词位均 `>= 0`。
- `NEAR` 里子区间的**区间**可以重叠，只有**实际使用的位置集合**必须两两不交。实现和测试都显式区分这两件事（见 `SemanticsTests.test_overlapping_child_intervals_disjoint_positions` 与 oracle）。
- 重复查询词、重复子树是不同需求：每个叶有独立见证槽，必须分到不同词位（如 `NEAR(3,[pump,pump])` → 用 0 和 2，不允许复用 0）。
- `SEQ` 的 gap 只数**未占用词位数**：相邻两词 `gap=0` 合法；`a x c` 需 `gap>=1`；`a b c` 用 `SEQ(1,[a,c])` 合法而 `SEQ(0,[a,c])` 不合法。

### 根节点最小窗口

只在根节点做：

1. 相同区间（start, end）只交付一个，按叶顺序的见证元组取字典序最小者（未使用叶按 `-1` 比较）。
2. 若同文档另有合格根区间**严格包含**于当前区间，删除较大的那个（含同起点的严格更长区间）。
3. 内部节点**不**按区间去重、不选最短区间：同一区间上的不同用词见证都必须保留，否则兄弟节点的互斥条件可能把唯一可行的见证丢掉（`NestedWitnessTests` 专门覆盖）。

### 输出

`search()` 每页返回 `Window` 列表，每项包含：

- `doc_id`、闭词位区间 `start`/`end`、`width`
- 完整 `witness`（按叶编号，未使用为 `-1`）
- `leaves`：每个叶的 token、原始拼写、实际词位列表、是否使用
- `char_span`、`byte_span`：首 token 起点到末 token 终点的原文跨度
- `snippet`：直接从落盘原文 `text[char_start:char_end]` 切出，**不**由 token 重新拼接

## 存储模式

| 表 | 内容 |
| --- | --- |
| `meta` | schema 版本、index_id、名称、指纹、文档/词位计数、代数 |
| `documents` | `doc_id` 主键、原始 Unicode 文本、`text_sha256`、字符/字节长度、token 数、该文档倒排行摘要、来源路径与来源 sha256 |
| `postings` | `(token, doc_id, pos)` 主键；`char_start/char_end/byte_start/byte_end` 列；`WITHOUT ROWID`；另有 `(doc_id, pos)` 索引 |

`postings` 就是落盘倒排词位表；原文只存一份，索引文件和正文绑定在同一 SQLite 文件里，切开也自包含。

### 正文与索引的绑定

- 每篇文档的内容摘要 `text_sha256` 和其倒排行的 `postings_digest` 都记录在 `documents` 上，整体再合成 `index_fingerprint`。
- `Index.verify(full=True)`：`PRAGMA quick_check`、重新分词、重算摘要、逐行比较倒排，并检查来源文件是否仍与 `source_sha256` 一致。篡改正文或倒排都会报出问题。
- `Index.check_source(doc_id)` / `open_index(..., verify=True)` 的默认检查**不**因来源文件变化而拒绝打开（已落盘正文仍是被索引的内容），但 `verify(check_sources=True)` 会明确报告“来源已变”。
- 换内容必须显式 `rebuild_document(index, doc_id, new_text, ...)`；它会原子地替换正文与倒排、更新指纹。**不存在**“同名文件自动重指”的路径。

## 候选生成与剪枝（为什么不扫全库）

1. 查询树先归约为**需求组**：`SEQ`/`NEAR` 的子需求取合取；`OR` 下所有叶构成一个析取组。任何命中文档必须满足每个组，否则不可能有根命中。
2. SQLite 在 `postings` 主键上对这些组做 `INTERSECT`，按 `doc_id` 排序返回候选。缺失任一必需词的文档不会进入求解器，其正文永远不会被读取。
3. 只加载候选文档中**查询叶 token** 的倒排行；不取全库正文。
4. 逐文档求解；最小窗口过滤严格在文档内，输出顺序固定为 `doc_id, start, end`，所以凑够“本页 + 1 个窗口”即可停止，后面的文档只可能追加更小的顺序位置。
5. 只有**真正交付窗口**的文档才会读取正文并切片；为证明 `has_more` 而多求解的那篇文档不会被读正文（`AccessTrace.body_reads` 记录证据）。

`AccessTrace` 统计：候选文档数、实际求解文档数、倒排查询次数、取回倒排行数、正文读取的 `doc_id` 列表、被必需 token 过滤掉的文档数。`search(..., trace=t)` 可传入自己的 trace 对象；`SearchPage.trace` 也有同样的字典。

### 保留什么、何时剪枝

- 所有中间命中都保留**完整叶→词位指派**；相同指派是同一命中，可安全去重；相同区间但不同用词绝不去重。
- 安全剪枝只有：无倒排行的子节点直接判空；`SEQ` 中后一区间起点必须大于前一区间终点、空档数超 `gap` 则丢弃；`NEAR` 中实际用位相交或包围区间超 `width` 则丢弃；`OR` 仅去掉完全相同的指派。
- 内部节点不做“只留最短区间”。

## 分页

- `limit`/`offset` 直接分页；也可用 `cursor`。
- 后续页顺序固定为 `doc_id, start, end`，页间不重复、不丢失（测试用 `limit=1,2,3` 逐页拼接后与一次性大页做全量对比）。
- 游标里带 `offset`、`page_size`、查询签名和索引指纹：
  - 换查询 → 报 `CursorError`；
  - 期间发生 `rebuild_document`（指纹变化）→ 报 `CursorError`，而不是静默跳过/重复。
- 数据固定，不要求实时更新期间的分页一致性；显式重建会使旧游标失效。

## 模块

| 文件 | 职责 |
| --- | --- |
| `nearidx/lexer.py` | 固定词法、casefold、字符/字节跨度、查询 token 校验 |
| `nearidx/query.py` | 查询 AST、解析器、先序叶编号、序列化/签名、分页游标 |
| `nearidx/db.py` | 建库、schema、只读打开、倒排访问、完整性校验、显式重建 |
| `nearidx/solver.py` | 需求组候选、候选窗口驱动求解、根最小窗口、分页 |
| `nearidx/cli.py` | `build/search/snippet/stats/verify/rebuild` |
| `nearidx/factcheck.py` | 可执行的主例/嵌套例演示 |
| `nearidx/tests/oracle.py` | 独立暴力 oracle：枚举每个叶的所有词位分配后按公开规则筛选 |
| `nearidx/tests/test_semantics.py` | 47 个测试（含随机化差分测试） |
| `examples/scale_demo.py` | 大量无关文档下的访问证据与超小分页演示 |

## 测试覆盖

- 主例：`NEAR(2,[pump,pump])` 空；`NEAR(3,[pump,pump])` 恰为 `[0,2]`，用 0/2 两个词位；`SEQ(0,[pump,valve])` 只给 `[0,1]`。
- 大小写折叠（含 `ß`→`ss`）、重复词、重复子树、重叠子区间但词位不交、内部同区间多见证、兄弟冲突时保留另一见证、根最小窗口、唯一见证决胜（未使用叶 `-1` 参与字典序）。
- 空文档、纯分隔符文档、多字节来源切片（emoji + 非 ASCII）。
- 超小分页（1/2/3）、游标拒绝换查询与重建。
- 新进程 `python -I` 重开索引并查询。
- 篡改正文/倒排检测、来源文件变动的显式报错、显式重建后的重新绑定。
- 独立 oracle：固定小文本 × 固定查询集合、随机文本 × 随机查询树（150 组），同时对比**原始命中**与**根窗口**；另对根窗口线性选择做 200 组随机 O(n²) 交叉校验。
- 规模例：2000 篇无关文档，记录候选数、求解数、正文读取数。

## 已知边界

- 单个中间节点的部分命中数超过 `max_hits`（默认 2,000,000）会抛 `SolverLimitError`，提示收窄查询；不做静默截断。
- 分页游标不是“快照句柄”：它只校验查询与索引指纹，不支持数据在分页期间变化后继续旧序列。
- 未实现并发写；`rebuild_document` 使用 `BEGIN IMMEDIATE` 做单次原子替换，构建走临时文件 + 原子 rename。
