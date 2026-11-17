# nearidx — 落盘词位索引与结构化近邻查询器

纯 Python + SQLite 实现。构建时把每篇文档切成词位并落盘倒排表，查询时先从倒排表取候选文档，再在候选文档内部求解 `SEQ` / `NEAR` / `OR` 的可嵌套查询树，最后只对真正要交付的窗口读取正文并切出原文片段。不依赖任何全文搜索引擎，不做分词模型、词干、同义词或词表训练。

## 快速开始

```bash
python3 -m nearidx build demo.nearidx docs/*.txt --overwrite
python3 -m nearidx search demo.nearidx 'NEAR(3,[pump,pump])' --limit 5
python3 -m nearidx fsearch vendors.json 'NEAR(3,[pump,pump])' --limit 5
python3 -m nearidx snippet demo.nearidx d1 --positions 0:2
python3 -m nearidx verify demo.nearidx --full
python3 -m nearidx rebuild demo.nearidx d1 --file docs/d1.txt
python3 -m nearidx refresh demo.nearidx batch.jsonl --retain-old 2
python3 -m nearidx archive demo.nearidx demo.nearidx.pkg
python3 -m nearidx inspect-archive demo.nearidx.pkg
python3 -m nearidx restore demo.nearidx.pkg restored.nearidx
python3 -m nearidx recover-restore restored.nearidx
```

Python API：

```python
from nearidx import build_index, open_index, parse_query, refresh_index, search

build_index("demo.nearidx", [("d1", "pump valve pump")], overwrite=True)
with open_index("demo.nearidx") as idx:
    page = search(idx, parse_query("NEAR(3,[pump,pump])"), limit=10)
    for w in page.windows:
        print(w.doc_id, w.start, w.end, w.witness, w.snippet, w.leaves)
```

联邦检索（一条查询同查多份索引）：

```python
from nearidx import Federation, federated_search, parse_query

with Federation({"vendor-a": "a.nearidx", "vendor-b": "b.nearidx"}) as fed:
    page = federated_search(fed, parse_query("NEAR(3,[pump,pump])"), limit=10)
    for w in page.windows:  # 按 (doc_id, source_name, start, end) 全局排序
        print(w.source_name, w.doc_id, w.start, w.end, w.witness, w.snippet)
    while page.has_more:  # 游标跨进程、跨刷新续翻各自固定的代
        page = federated_search(fed, parse_query("NEAR(3,[pump,pump])"),
                                cursor=page.next_cursor)
```

运行测试与演示：

```bash
python3 -m unittest discover -s nearidx/tests -t .   # 166 个测试
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

批量刷新发布的每个旧代都是同样自包含的完整索引文件（`INDEX.gen-<N>`），由 `INDEX.generations.json` 登记；当前代始终是 `INDEX` 本身。

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
- 游标里带 `offset`、`page_size`、查询签名和**索引代指纹**：
  - 换查询 → 报 `CursorError`；
  - 无游标的第一页绑定开始查询时的索引代；它返回的后续游标继续指向**同一代**，而不是每次翻页时重新指向当前代。旧代仍保留时，旧游标跨进程、跨重开都继续看到原文、原命中顺序和原分页边界；
  - 旧代已被清理（超出保留范围）→ 报 `GenerationExpiredError`（`CursorError` 的子类），绝不静默改查当前代；
  - 期间发生 `rebuild_document`（原地改写，不保留旧代）→ 同样报错失效。

## 批量刷新（代际发布）

`nearidx refresh INDEX MANIFEST.jsonl` 把一整批审核变更发布为一个新索引代；Python 侧对应 `refresh_index(path, manifest, retain_old=...)`。清单是 JSON Lines，每行一项操作：

```jsonl
{"op":"upsert","doc_id":"d1","text":"新正文"}
{"op":"upsert","doc_id":"d2","path":"docs/d2.txt"}
{"op":"delete","doc_id":"d3"}
```

- `upsert` 必须且只能提供 `text` 或 `path`；`delete` 只带 `doc_id`。新增与替换同用一个 `upsert`。
- 同一批次中同一 `doc_id` 只能出现一次；删除不存在的文档、坏 JSON、无法读取或解码的来源、重复 ID、未知操作/字段都**拒绝整批**——所有校验和来源读取都在任何文件改动之前完成，失败时当前索引与进行中的分页保持原样。
- 成功批次作为一个新代发布：新内容先构建为完整索引文件，写入 `pending` 日志后经两次原子 rename 换入，返回新代号与指纹。发布期间进程被杀，重开时 `recover_generations`（每次打开都会运行）只会恢复出**完整的旧代或完整的新代**；未发布的孤立文件永远不会被当成当前代。
- 旧代以 `INDEX.gen-<N>` 完整文件保留，由 `INDEX.generations.json` 登记。`--retain-old N`（持久化在注册表里）控制保留多少个旧代：默认保留当前代及最近两个旧代，最少必须保留一个旧代；超出保留范围的代文件被清理，对应游标报过期。
- 一个 `Index` 句柄始终服务它打开时的那一代；要看新代请重新 `open_index`。`idx.generations()` 列出当前代与所有保留代，`idx.verify(full=True, generations=True)`（CLI `verify --full --generations`）会逐代校验数据与指纹绑定。
- 与 `rebuild_document` 的区别：单篇重建仍是原地原子替换、使旧游标失效；批量刷新才发布新代并保留旧代。`build_index(overwrite=True)` 是全新索引，会丢弃上一代际状态。

## 联邦检索（多来源同查）

审核场景里同一条 `NEAR`/`SEQ` 查询常常要同时查阅多个供应商批次。`nearidx fsearch MANIFEST.json QUERY`（Python 侧 `Federation` + `federated_search`）把若干份**互相独立**的索引当成一个整体来查：清单是稳定来源名到索引文件的映射，结果按 `(doc_id, source_name, start, end)` 全局排序分页，每个窗口保留来源名与原有的窗口/见证/原文跨度。不同来源出现相同 `doc_id` 是**不同的结果**，全部保留。

清单是一个 JSON 对象（相对路径相对清单所在目录解析）：

```json
{"vendor-a": "indexes/a.nearidx", "vendor-b": "indexes/b.nearidx"}
```

- **只读**：联邦检索只通过 `open_index` 只读打开各来源，从不修改任何索引内容。任一来源缺失、格式无效或清单非法（重复来源名、空清单、非字符串路径……）都会让整次调用明确失败，绝不悄悄跳过来源后返回看似完整的部分结果。
- **代际固定**：第一页开始时分别固定每个来源实际打开的索引代；返回的游标逐来源记录索引身份（`index_id`）、固定代指纹和续翻位置。之后的游标跨进程、跨重开都继续读这些代——即使其中一份或多份索引已经 `refresh` 出新代。任一来源的固定代被清理出保留范围时，整次调用报 `GenerationExpiredError`，不给部分结果。
- **游标绑定**：游标同时绑定查询、页大小、来源名集合与每个来源的索引身份。更换来源清单（增删来源）、把来源名指向另一索引、或更换查询，旧游标都被明确拒绝（`CursorError`）；游标自带页大小，续翻时忽略新的 `--limit`。
- **流式归并**：每个来源的窗口流本身就按全局键有序，联邦层做 k 路归并——凑满当前页外加判定 `has_more` 所需的一个窗口即返回，绝不先把所有来源的全部命中读入内存再排序；续翻页同样只保留本页窗口和逐来源的续翻位置（每页每来源至多重解一个边界文档）。正文只在窗口真正交付时读取。
- **语义一致**：窗口由与单索引 `search` 完全相同的求解器、根最小窗口、见证与原文切片机制产生；测试用单索引结果逐项对照（含 `OR` 未使用叶、多字节原文跨度）。

每页的 `sources` 字段记录本页实际读取的逐来源 `{path, index_id, generation, fingerprint}`，`trace` 字段给出逐来源的访问证据（候选数、求解数、正文读取列表），便于审计"没有跨代混合、没有整库收集"。

## 归档与恢复（机器迁移 / 故障恢复）

`nearidx archive INDEX ARCHIVE`（Python：`archive_index(index, archive)`）把一份索引当前代、登记表中**全部**仍保留的旧代，以及 `INDEX.generations.json` 本身打成一个自包含归档，并附一份版本化清单：每个成员记录逻辑用途（`current` / `generation` / `registry`）、代号、指纹、原文件名、载荷偏移、大小和 SHA-256；清单同时绑定 `index_id`、当前代号与指纹、保留代列表和 `retain_old`。

- **流式**：成员按固定块（`CHUNK_SIZE`）边复制边计算长度与 SHA-256，归档与恢复都不会把成员文件整体读入内存；清单在所有载荷之后写出，文件头/尾固定结构 + 清单摘要一起绑住整份容器，缺成员、多成员、插入字节都会被发现。
- **只对应一个一致的代际登记状态**：归档前快照登记表字节与每个成员的文件签名（inode/size/mtime），复制后再核对；归档期间若发生 `refresh`、`rebuild_document` 或登记表改写，归档明确失败并删除未发布的临时产物。登记表与当前文件不一致、旧代文件缺失/是符号链接、存在未完成的 `pending` 发布日志，同样直接失败。
- 归档只包含索引自身（正文已在 `documents` 表内）；**外部来源文件不打包**，恢复也不会改写 `source_path` / `source_sha256`，来源文件缺失只在 `verify(check_sources=True)` 中作为“来源不可读”报告。

`nearidx restore ARCHIVE INDEX`（Python：`restore_index(archive, index)`）只恢复到**尚不存在**的新路径。校验顺序是：容器结构与版本 → 清单版本/成员清单/偏移覆盖 → 每个成员载荷长度与 SHA-256（对实际字节流式计算）→ 归档内的代际登记表与清单逐项一致 → 逐个成员打开为 SQLite 做 `quick_check`、重新分词比对正文/倒排摘要、核对 `index_id`、代号与指纹。全部通过后才安装文件；旧代先落位、代际登记表随后写入、**当前代最后**原子 rename 到目标路径。原文件名被映射为新目标名（`NEW.gen-<N>`），身份、代号和指纹保持不变，因此归档前签发的单索引游标与联邦游标在新文件名下继续可用（含跨进程）。

拒绝的情况：路径穿越 / 绝对路径 / 目录分隔符 / 带 NUL 的成员名、重复或未登记成员、符号链接、缺失或多余成员、截断容器、剥离/替换载荷字节、损坏 SQLite、身份/指纹/代号不符、不支持的清单或容器版本、目标已存在（含其登记表/旧代文件）。任何拒绝都不会留下可打开的半成品，也不会覆盖已有索引或改动源索引。

安装过程由目标目录内的标记文件记录：中途被杀时，目标路径要么完全不存在、要么已经是完整索引（当前代最后 rename）；再次 `restore` 或 `nearidx recover-restore INDEX` 会完成（当前代已就位且各代一致时补写登记表并清理）或整份丢弃中途产物。恢复重试总能在同一路径得到完整索引。

```python
from nearidx import archive_index, inspect_archive, restore_index

archive_index("demo.nearidx", "demo.nearidx.pkg")
m = inspect_archive("demo.nearidx.pkg")     # 流式校验载荷摘要
print(m.index_id, m.current_generation, [g for g, _ in m.retained])
restore_index("demo.nearidx.pkg", "restored.nearidx")
```

## 模块

| 文件 | 职责 |
| --- | --- |
| `nearidx/lexer.py` | 固定词法、casefold、字符/字节跨度、查询 token 校验 |
| `nearidx/query.py` | 查询 AST、解析器、先序叶编号、序列化/签名、分页游标 |
| `nearidx/db.py` | 建库、schema、只读打开、倒排访问、完整性校验、显式重建、代际注册表与崩溃恢复 |
| `nearidx/refresh.py` | 清单解析/校验、整批发布为新索引代、保留清理 |
| `nearidx/solver.py` | 需求组候选、候选窗口驱动求解、根最小窗口、分页 |
| `nearidx/federate.py` | 联邦清单、来源打开、跨来源全局归并分页与游标绑定 |
| `nearidx/archive.py` | 流式归档/恢复：代际一致快照、版本化清单、载荷摘要校验、崩溃安全安装与恢复日志 |
| `nearidx/cli.py` | `build/search/fsearch/snippet/stats/verify/rebuild/refresh/archive/inspect-archive/restore/recover-restore` |
| `nearidx/factcheck.py` | 可执行的主例/嵌套例演示 |
| `nearidx/tests/oracle.py` | 独立暴力 oracle：枚举每个叶的所有词位分配后按公开规则筛选 |
| `nearidx/tests/test_semantics.py` | 47 个语义测试（含随机化差分测试） |
| `nearidx/tests/test_refresh.py` | 37 个批量刷新测试：代际发布、游标绑定、保留清理、崩溃恢复 |
| `nearidx/tests/test_federate.py` | 40 个联邦检索测试：全局归并、代际固定、游标拒绝、流式证据 |
| `nearidx/tests/test_archive.py` | 42 个归档恢复测试：成员组成、游标续翻（含跨进程、联邦）、篡改/穿越/符号链接/版本拒绝、流式证据、中断重试 |
| `examples/scale_demo.py` | 大量无关文档下的访问证据与超小分页演示 |

## 测试覆盖

- 主例：`NEAR(2,[pump,pump])` 空；`NEAR(3,[pump,pump])` 恰为 `[0,2]`，用 0/2 两个词位；`SEQ(0,[pump,valve])` 只给 `[0,1]`。
- 大小写折叠（含 `ß`→`ss`）、重复词、重复子树、重叠子区间但词位不交、内部同区间多见证、兄弟冲突时保留另一见证、根最小窗口、唯一见证决胜（未使用叶 `-1` 参与字典序）。
- 空文档、纯分隔符文档、多字节来源切片（emoji + 非 ASCII）。
- 超小分页（1/2/3）、游标拒绝换查询与重建。
- 批量刷新：新增/替换/删除同批发布；旧游标跨重开、跨进程读完原代；无效批次整批拒绝且索引不变；保留代清理后旧游标报过期；发布在每一步被杀后恢复为完整旧代或新代；单篇重建与普通查询行为不变。
- 联邦检索：多来源交错（含相同 doc_id）的全局顺序与每页边界；第一页后只刷新一个索引，旧游标跨进程读完各自原代、新查询看到当前代；更换来源映射、重复来源名、缺失/损坏索引、任一来代际过期都整次拒绝；1500 篇全命中文档上首页只求解 ≤5 篇文档；联邦窗口与单索引 `search` 逐项一致。
- 新进程 `python -I` 重开索引并查询。
- 归档恢复：无旧代 / 单个 / 多个旧代（含 `retain_old` 裁剪后）的成员组成与清单字段；归档前签发的单索引游标与联邦游标在恢复到**不同文件名**后、在新进程里继续逐页比对窗口/来源/顺序/见证/原文片段；新查询读恢复后的当前代；归档期间登记表或当前文件被改写 → 明确失败且无产物；成员被替换（字节级）、截断、遗漏、额外插入、重复/未登记、恶意路径、替换为另一份合法索引、清单/容器版本不支持、身份/指纹不符、目标已存在、符号链接均被拒绝且目标与源索引不变；恢复在每次 rename 前后被杀后目标要么不可打开、要么完整可用，重试可完成；归档/恢复的每次读取都以 `CHUNK_SIZE` 为上限流式进行。
- 篡改正文/倒排检测、来源文件变动的显式报错、显式重建后的重新绑定。
- 独立 oracle：固定小文本 × 固定查询集合、随机文本 × 随机查询树（150 组），同时对比**原始命中**与**根窗口**；另对根窗口线性选择做 200 组随机 O(n²) 交叉校验。
- 规模例：2000 篇无关文档，记录候选数、求解数、正文读取数。

## 已知边界

- 单个中间节点的部分命中数超过 `max_hits`（默认 2,000,000）会抛 `SolverLimitError`，提示收窄查询；不做静默截断。
- 分页游标绑定一个索引代：该代仍保留时可跨进程续翻；被清理后报 `GenerationExpiredError`。游标不是”快照句柄”之外的实时同步机制，`rebuild_document` 原地改写仍使旧游标失效。
- 联邦游标绑定的是”查询 + 页大小 + 来源名集合 + 每个来源的索引身份与固定代”：来源清单或查询有任何变化都会被拒绝；联邦层自身不持有跨调用的状态，每次翻页按游标重开各来源的固定代。
- 未实现并发写：`rebuild_document` 使用 `BEGIN IMMEDIATE` 做单次原子替换，构建走临时文件 + 原子 rename，批量刷新走完整新文件 + 日志 + 原子 rename；崩溃恢复假设没有另一个写者同时进行（恢复无法区分“崩溃的发布”和“进行中的发布”）。
- 批量刷新按整库构建新代文件，发布成本随语料大小线性增长；换来的是发布与恢复路径完全复用构建逻辑。
- 归档一致性依赖“归档期间没有并发写者把同一代原地改写却保持大小与 mtime 不变”——`refresh` 的两次 rename 与 `rebuild_document` 都会改变文件签名；归档不持有 SQLite 锁，也不做跨进程快照。归档只携带登记表当时登记的代，更早被裁剪的代不会再出现。
- 归档格式是单文件容器（`nearidx-archive` v1），成员以平铺文件名存储；清单版本、容器版本与索引 `schema_version` 各自独立校验，未来升级时旧版本会被明确拒绝而不是猜测解析。
