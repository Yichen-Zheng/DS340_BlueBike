# DS340_BlueBike

## 第一阶段 · 可用还车空位回归

给定站点和基准时间 t，预测 t+15 分钟的可用空位数量。仓库现有的 `Bluebike.ipynb` 读取行程数据，没有站点列表前端。可通过下方 CLI 或 Python 的 `predict(artifact, db, station_id, at, horizon=15)` 接入站点列表。仅使用 `status == "ok"` 的真实数据预测，展示为“预测空位”及对应的 `target_time`。拒绝还车或未安装的站点不会产生可用预测。

### 数据检查

| 本地文件 | 行程数 | 出发时间范围 | 到达时间范围 | 非空站点 ID 数（起终点并集） |
| --- | ---: | --- | --- | ---: |
| 2607.csv | 530,297 | 2026-06-30 04:33 至 07-31 23:56 | 07-01 00:00 至 07-31 23:59 | 606 |
| 2608.csv | 582,663 | 2026-07-31 01:38 至 08-31 23:56 | 08-01 00:00 至 08-31 23:59 | 617 |
| 2609.csv | 555,205 | 2026-08-30 23:38 至 09-30 23:54 | 09-01 00:00 至 09-30 23:59 | 621 |

时间按文件原始值显示到分钟，CSV 没有时区偏移。字段包括行程 ID、车型、起终时间、起终站名称和 ID、起终点经纬度、会员类型。每行是一笔骑行事件，采样频率不固定。行程记录缺少车桩状态、停用车桩和车辆调度，因此无法完整还原可用空位。训练和预测使用 GBFS 站点 ID。若站点列表使用行程 ID，需要另行验证 ID 映射。

[Bluebikes 官方数据页](https://bluebikes.com/system-data) 提供行程历史、含容量的站点清单及[实时 GBFS](https://gbfs.bluebikes.com/gbfs/gbfs.json)。项目检查未发现官方可下载的逐时刻站点状态历史。首次实采覆盖 595 个站点，195 条记录的可用空位与容量减可用车辆数不同。标签使用官方 `num_docks_available`，空值保留为缺失。

### 实时采集

采集器只用 Python 标准库。建模已在项目机的 Anaconda 环境验证，使用 scikit-learn 1.4.2、NumPy 1.26.4 和 joblib。下方 `python` 应指向该环境，例如本机可使用 `/opt/anaconda3/bin/python`。

```bash
python collect_status.py --once
python collect_status.py --db data/status.sqlite --interval 60
```

持续采集命令在前台运行，Ctrl+C 停止。需要跨终端或重启持续采集时，用系统的服务管理工具常驻运行同一命令。中断后可继续写入同一数据库。脚本会记录网络及数据错误，并在下一轮重试。

SQLite 保存采集时间、feed 更新时间、站点上报时间（UTC Unix 秒）、站点 ID、可用空位、可用车辆、安装/租车/还车状态、容量及位置。同站同秒的重复记录按主键去重。写入以整批快照为事务。默认每 60 秒拉取，此间隔是采集设定，实际匹配始终使用时间戳。采集器拒收超过 5 分钟的 feed。运行产生的数据库和模型输出由 Git 忽略。

### 训练、评估与预测

建议先积累覆盖工作日、周末和不同客流时段的连续真实快照。单批快照无法训练。

```bash
python dock_regression.py train --db data/status.sqlite --model artifacts/real-15.joblib --horizon 15
python dock_regression.py evaluate --model artifacts/real-15.joblib
python dock_regression.py predict --db data/status.sqlite --model artifacts/real-15.joblib \
  --station-id "$GBFS_STATION_ID" --at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --horizon 15
```

将 `GBFS_STATION_ID` 设为训练覆盖的站点 ID。基准时间接受带显式偏移的 ISO 8601。响应包括数据更新时间、站点上报时间、目标时间、连续预测值、模型版本和数据来源。无预测时返回空预测值和明确状态，例如 `unsupported_horizon`、`unknown_station`、`missing_history`、`stale_data`、`missing_dock_count` 或 `returns_unavailable`，CLI 退出码为 2。只加载自己生成或可信来源的 joblib 文件，并使用与训练相同的依赖版本。

- **匹配与特征**。标签取同站 t+15 分钟附近最近的真实观测，容差 ±90 秒，等距离取较早观测。从 t 到标签之间如有相邻记录间隔超过 180 秒则跳过。过去 5、15、30 分钟的特征只向后查找，允许最多早 120 秒，历史窗口同样排除长间隔。用于模型的当前和匹配观测要求 feed 与站点上报均在 5 分钟以内。时段、周几和周末特征按波士顿时区计算，另使用当前空位、5 分钟变化、站点 ID、容量和坐标。
- **评估**。按全部观测时间点的 60/20/20 顺序分割，剔除计划目标时间或实际标签时间越界的样本。验证、测试只评估训练覆盖的站点。站点编码、缺失值填补和缩放均只在训练集拟合。验证集 MAE 用于选择 Ridge 正则化强度，测试集用于最终评估。Persistence 与回归使用相同样本并按同一规则裁剪至非负和已知容量上限，保留小数。报告全量和实际空位 0–2 子集的 MAE、RMSE 和样本数。评估命令读取训练时保存的测试报告。
- **范围**。默认模型只用于 15 分钟预测。需要 5 或 30 分钟时，可分别训练并单独评估相应的 `--horizon` 模型。接入任意预计到达时间，需要验证更多时长或训练以时长为输入的模型。概率预测需要独立的目标、模型和验证。

### 模拟验证

下面生成的数据库和模型均明确标记为 `simulated`，只用于验证流程。

```bash
python -m unittest -v
python -c "from test_dock_regression import create_demo; create_demo('data/simulated.sqlite')"
python dock_regression.py train --db data/simulated.sqlite --model artifacts/simulated-15.joblib --allow-simulated
python dock_regression.py evaluate --model artifacts/simulated-15.joblib
python dock_regression.py predict --db data/simulated.sqlite --model artifacts/simulated-15.joblib \
  --station-id SIM-A --at 2026-07-01T18:00:00-04:00
```

真实快照历史尚不足，当前没有真实测试集 MAE 或 RMSE。

## 感恩节净流量对比与调整参数

使用 [Bluebikes 官方行程下载目录](https://s3.amazonaws.com/hubway-data/index.html) 的 2024、2025 年 7、8、9、11 月文件，以及 2024 年 12 月文件。原始文件保留在临时目录，未写入仓库。2024 年感恩节周为 11 月 25 日至 12 月 1 日，2025 年为 11 月 24 日至 11 月 30 日。

| 年份 | 感恩节周出发数 | 同年 11 月普通周参考值 | 同年 7–9 月参考值 |
| --- | ---: | ---: | ---: |
| 2024 | 49,816 | 101,954 | 126,916 |
| 2025 | 52,134 | 90,830 | 127,734 |

周参考值为逐星期几平均后求和，四舍五入显示。11 月参考周从 2024 年 11 月 4 日、18 日及 2025 年 11 月 3 日、17 日开始，避开退伍军人节所在周。夏季参考排除 7 月 4 日和劳动节。净流量使用感恩节周及普通参考的同一站点集合，取当年 11 月和夏季参考均有行程事件的站点，2024 年为 505 个，2025 年为 581 个。每站每天使用本地时间对齐的 96 个 15 分钟窗口。每段净流量等于结束于该站的行程数减去开始于该站的行程数。没有行程事件的窗口按零流量计入，数据无法判断站点当时是否开放。

| 感恩节当天测试集 | 站点窗口数 | 夏季原始 MAE / RMSE | 11 月普通周四 MAE / RMSE | 夏季乘另一年拟合的系数 MAE / RMSE |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 48,480 | 0.277 / 0.627 | 0.330 / 0.742 | 0.066 / 0.302 |
| 2025 | 55,776 | 0.285 / 0.606 | 0.350 / 0.761 | 0.124 / 0.446 |

误差单位为辆每站每 15 分钟。将夏季同站、同星期几、同时段的平均净流量乘以 `holiday_factor`。在一年感恩节当天用 MAE 从 0 到 1.5（步长 0.05）选系数，再拿另一年感恩节当天检验。两次都选出 0，因此表中调整后结果等于零净流量基线。该结论仅适用于本次检验的对齐窗口和均值基线。节前、节后、任意分钟出发和真实空位精度均需额外验证。系数默认值为 1，运行时可显式指定。系数为 0 时，空位估算沿用当前值。

```bash
python holiday_flow.py compare --data-dir /tmp/bluebikes-holiday > artifacts/holiday-flow.json
python holiday_flow.py estimate --current-docks 6 --expected-net-returns 4 --capacity 20 --holiday-factor 0.5
```

比较命令读取官方原名的 CSV 或 ZIP，输出完整的逐日数据和交叉年度测试结果。估算接口接受已知的当前可用空位和外部提供的净还车预测，按 `当前空位 − 系数 × 净还车预测` 计算并限制在可用范围内。仅对新鲜、允许还车的快照调用。做 GBFS 验证时，对同一站点和预测时段保留系数 0、1 以及任何待比较的候选值，用 15 分钟后的官方空位计算 MAE、RMSE，并单列实际空位 0–2 的样本。

同样把周一到周日全部用于拟合，在另一年的完整感恩节周测试。两次拟合也都选出系数 0。

| 整周测试集 | 站点窗口数 | 夏季原始 MAE / RMSE | 夏季乘整周系数 MAE / RMSE |
| --- | ---: | ---: | ---: |
| 2024 年感恩节周 | 339,360 | 0.336 / 0.709 | 0.196 / 0.590 |
| 2025 年感恩节周 | 390,432 | 0.313 / 0.645 | 0.186 / 0.562 |

整周系数在周四子集上的结果也与上方周四系数一致。整周与周四结果分别位于 JSON 的 `whole_week` 和 `thursday` 中。整周结果按站点与窗口汇总，不能代表每一天都分别选出了相同系数。

## 普通周真实空位验证

固定测试期为波士顿时间 2026 年 9 月 14–27 日。真值使用档案中的 `free` 字段。历史净流量只用 `2607.csv` 和 `2608.csv`，排除 7 月 4 日，按站点、星期几和一天中的分钟取平均。以观测后第一个整分钟作为历史窗口起点，合计接下来 15 分钟。系数 0 沿用当前空位，系数 1 减去历史净还车数并按当前容量裁剪。两种方法使用同一组测试样本，未在测试集上调参。

用官方站点清单的 `station_id → short_name` 字段把 CityBikes `nuid` 与行程站号连接，保留唯一映射和拥有历史流量的站点。选取 ±90 秒内最近的同站目标状态。两端都要求状态上报在 5 分钟内、为实体站点、允许还车且空位在容量范围内。排除跨周或者观测间隔超过 3 分钟的目标窗口。该档案为不规则的状态变化记录，严格筛选会偏向活跃站点和时段，相邻窗口也可能重叠，因此结果仅适用于合格样本。

下载 [CityBikes 2026 年 9 月档案](https://data.citybik.es/dumps/by-network/2026/202609-blue-bikes-stats.parquet) 和 [Bluebikes 官方站点清单](https://gbfs.lyft.com/gbfs/1.1/bos/en/station_information.json) 后，可在安装了 pandas 和 PyArrow 的建模环境运行。官方站点清单为运行时下载的当前版本，未来重新下载可能改变覆盖。

```bash
mkdir -p artifacts
python validate_ordinary_weeks.py \
  --archive /tmp/202609-blue-bikes-stats.parquet \
  --station-information /tmp/bluebikes-station-information.json \
  --trips 2607.csv 2608.csv > artifacts/ordinary-weeks.json
```

Bike-share data by CityBikes contributors and available from https://data.citybik.es.

## 真实站点快照回归 · 2026 年 9 月

使用上节同一个 CityBikes Parquet 档案。导入脚本以 `nuid` 为站点 ID、`free` 为可用空位、`extra.last_updated` 为站点上报时间。档案缺少独立的 feed 更新时间和安装标记，因此使用档案采集时间作为 feed 更新时间，只导入明确标为实体的站点，并以 `returning=1` 判断是否接受还车。档案与实时 GBFS 使用各自的 SQLite 文件。

```bash
python import_citybikes.py --archive /tmp/202609-blue-bikes-stats.parquet \
  --db data/citybikes-202609.sqlite
python dock_regression.py train --db data/citybikes-202609.sqlite \
  --model artifacts/citybikes-real-15.joblib \
  --validation-start 2026-09-10T00:00:00-04:00 \
  --test-start 2026-09-14T00:00:00-04:00 \
  --test-end 2026-09-28T00:00:00-04:00 > artifacts/citybikes-real-15.json
python dock_regression.py evaluate --model artifacts/citybikes-real-15.joblib
python dock_regression.py predict --db data/citybikes-202609.sqlite \
  --model artifacts/citybikes-real-15.joblib \
  --station-id 2012646293885032702 --at 2026-09-14T21:18:48Z
```

训练取 9 月 1–9 日，验证取 9 月 10–13 日，测试取 9 月 14–27 日，均按波士顿时间划分。训练 384 个样本。验证集从三个 Ridge 正则化强度中选出 100。测试只有 755 个样本、10 个训练覆盖站点，因为模型要求 5、15、30 分钟的历史特征及密集观测。这与上一节只需当前和目标快照的样本范围不同。

| 测试期（按预测基准时间） | 样本数 | 沿用当前空位 MAE / RMSE | Ridge MAE / RMSE |
| --- | ---: | ---: | ---: |
| 9 月 14–20 日 | 428 | 4.079 / 5.104 | 3.632 / 4.847 |
| 9 月 21–27 日 | 327 | 2.844 / 3.854 | 3.412 / 4.249 |
| 合计 | 755 | 3.544 / 4.604 | 3.537 / 4.598 |

实际空位为 0–2 的 61 条样本中，baseline 和 Ridge 的 MAE 分别为 4.902 和 5.309。以预测至少 1 个空位作为可还车判断，两种方法分别把 20 次实际满站中的 16 次和 20 次误报为有空位。保留 persistence 作为当前基线。报告未加入学期变量，单月数据只覆盖一次开学过渡，无法验证该变量能否跨学期提升预测。

## 缺失历史特征与 10 月独立测试

通过 `--allow-missing-lags` 训练可处理历史缺失的 Ridge。5、15、30 分钟的历史空位分别加入缺失标记和观测距离基准时间的秒数。找不到满足容差及新鲜度的观测，或观测位于超过 3 分钟的缺测间隔之前，便将该历史值视为缺失。填补器只在训练集拟合。允许缺失仅影响历史特征，当前观测的新鲜度、可还车状态、目标时间匹配与缺测限制保持一致。既有模型保留原有完整历史要求。

数据来自 [CityBikes 2026 年 9 月档案](https://data.citybik.es/dumps/by-network/2026/202609-blue-bikes-stats.parquet) 与 [10 月档案](https://data.citybik.es/dumps/by-network/2026/202610-blue-bikes-stats.parquet)。将两个文件导入同一个独立档案库。以波士顿时间划分 9 月 1 日至 10 月 4 日为训练期，10 月 5–18 日为验证期，10 月 19–25 日为此前未用于选择模型的最终测试期。验证集选 Ridge 正则化强度。两种方法在同一批测试样本上计算误差，预测值保留小数。

```bash
mkdir -p artifacts
python import_citybikes.py --archive /tmp/202609-blue-bikes-stats.parquet \
  --db data/citybikes-202609-10.sqlite
python import_citybikes.py --archive /tmp/202610-blue-bikes-stats.parquet \
  --db data/citybikes-202609-10.sqlite
python dock_regression.py train --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-optional-lags-15.joblib --allow-missing-lags \
  --validation-start 2026-10-05T00:00:00-04:00 \
  --test-start 2026-10-19T00:00:00-04:00 \
  --test-end 2026-10-26T00:00:00-04:00 > artifacts/citybikes-optional-lags-15.json
python dock_regression.py evaluate --model artifacts/citybikes-optional-lags-15.joblib
```

实际训练使用 30,994 条样本，验证集选出 `alpha=1`。独立测试中两种方法都使用 5,057 条样本、144 个站点。其中同时具备三项历史特征的只有 381 条、10 个站点。档案按不规则时间记录状态变化，测试仍偏向目标期间记录密集的站点和时段。

| 测试指标 | 沿用当前空位 | 允许缺失历史的 Ridge |
| --- | ---: | ---: |
| 全量 MAE / RMSE | 3.728 / 4.886 | 3.635 / 4.649 |
| 实际 0–2 空位 MAE / RMSE（498 条） | 3.813 / 5.351 | 4.633 / 5.609 |
| 实际满站却估算至少有 1 个空位（127 条） | 119 | 127 |
| 实际至少 1 个空位却估算不足 1 个 | 80 | 8 |

整体误差下降并未改善近满站表现。保留「沿用当前空位」作为对照基线。目前两种方法的满站误报都很高，尚不适合作为有位可还的依据。

## 满站筛选门槛

`predict` 可通过 `--minimum-recommendation-docks` 调整站点通过推荐筛选所需的最少预测空位。预测仍是连续空位数。达到门槛时 `meets_return_threshold=true`，低于门槛时为 false。没有有效预测时为 null。未校准模型默认门槛为 1，校准后的模型默认使用保存的门槛，CLI 可手动覆盖。

下面保留已有回归模型，用 10 月 5–18 日验证集选择门槛。以 0.5 个空位为步长，选取最小的门槛，使验证集中实际满站样本通过筛选的比例不超过 50%。这是用于比较的可调目标。再用 10 月 26–30 日留出的 4,386 条归档样本验证。保存带门槛的独立模型文件，回归预测值保持原样。

```bash
python dock_regression.py calibrate-return-threshold \
  --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-optional-lags-15.joblib \
  --output-model artifacts/citybikes-threshold-15.joblib \
  --test-start 2026-10-26T00:00:00-04:00 --test-end 2026-10-31T00:00:00-04:00 \
  --max-full-station-false-open-rate 0.5 > artifacts/citybikes-threshold-15.json
python dock_regression.py predict --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-threshold-15.joblib \
  --station-id 00802fe6-4268-4dc8-8f55-0dbbc1665ce6 --at 2026-10-19T04:04:22Z \
  --minimum-recommendation-docks 9
```

| 留出测试 | 实际满站却通过门槛（共 157 条） | 实际有空位却未通过（共 4,229 条） |
| --- | ---: | ---: |
| Ridge，原门槛 1 | 156 | 19 |
| Ridge，校准门槛 5.5 | 52 | 783 |
| 沿用当前空位，校准门槛 4.5 | 59 | 802 |

筛选门槛适用于比较漏掉可还站点和误推满站的取舍。测试比例只描述所用的历史样本。

## 按历史借还车流量给站点分组

探索脚本用 2026 年 7–8 月行程，排除 7 月 3 日和 4 日，按工作日/周末、四个时段以及借车/还车计算流量。对各站的流量规模和时段方向分布标准化后，用 K-means 划分四组。官方 GBFS 清单只用于连接站点 ID。历史不足或无法匹配的站点沿用全城模型。每组至少有 500 个训练样本且覆盖 5 个训练站点才单独拟合 Ridge。

```bash
python flow_type_experiment.py --trips 2607.csv 2608.csv \
  --station-information /tmp/bluebikes-station-information.json \
  --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-optional-lags-15.joblib \
  --classifier artifacts/citybikes-flow-classifier.joblib \
  > artifacts/citybikes-flow-types-exploratory.json
```

保存的分组模型包含特征标准化器、聚类中心和已分组站点的行程 ID。2026 年 10 月 5–18 日共 10,408 条相同验证样本上，全城模型和流量分组模型的全量 MAE 分别为 3.418 和 3.349，实际只有 0–2 个空位时的 MAE 分别为 4.645 和 4.867。该结果为探索性验证，现用回归模型和还车门槛保持原值。

### 在全城 Ridge 中加入流量组

`train` 可以用 `--flow-classifier` 和 `--station-information` 将已保存的流量组作为一个类别特征。所有站点共享同一模型。缺少流量组但有训练数据的站点使用 `unknown` 类别。训练后的模型内含站点到流量组的映射，推理无需再读取分类器或站点清单。

```bash
python dock_regression.py train --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-pooled-flow-group-15.joblib --allow-missing-lags \
  --flow-classifier artifacts/citybikes-flow-classifier.joblib \
  --station-information /tmp/bluebikes-station-information.json \
  --validation-start 2026-10-05T00:00:00-04:00 \
  --test-start 2026-10-19T00:00:00-04:00 --test-end 2026-10-26T00:00:00-04:00 \
  > artifacts/citybikes-pooled-flow-group-15.json
```

和旧模型在相同的 10 月 19–25 日样本上做诊断比较。5,057 条样本中有 4,971 条匹配到流量组，覆盖 144 个站点中的 135 个。两版全量 MAE（四舍五入）均为 3.635，近满站 MAE 均为 4.633。以 1 个预测空位为门槛，两版在 127 个满站样本上均误报有空位。这个已使用过的测试周用于同样本对照，仍需新的留出期验证该特征。候选模型独立保存，已校准门槛保留在原来的模型文件。

## 空位变化量回归与同覆盖率比较

`--predict-change` 将回归目标设为 15 分钟后的空位减当前空位。推理时把预测变化加回当前空位，再限制在非负和容量范围内。模型同时加入最近一次可见的空位变化、该变化首次被观测到距现在多久，以及变化和流量组的交互项。特征仅使用基准时间及以前的状态，最多回看约半小时。历史间隔超限、数据不新鲜或无法观察到变化时标记缺失。

```bash
python dock_regression.py train --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-dock-change-15.joblib --allow-missing-lags --predict-change \
  --flow-classifier artifacts/citybikes-flow-classifier.joblib \
  --station-information /tmp/bluebikes-station-information.json \
  --validation-start 2026-10-05T00:00:00-04:00 \
  --test-start 2026-10-19T00:00:00-04:00 --test-end 2026-10-26T00:00:00-04:00 \
  > artifacts/citybikes-dock-change-15.json

python compare_return_coverage.py --db data/citybikes-202609-10.sqlite \
  --direct-model artifacts/citybikes-pooled-flow-group-15.joblib \
  --change-model artifacts/citybikes-dock-change-15.joblib \
  --start 2026-10-19T00:00:00-04:00 --end 2026-10-26T00:00:00-04:00 \
  > artifacts/citybikes-change-coverage.json
```

比较对三种方法使用相同的站点和观察时间。按预测空位从高到低排序，在全部样本上各取相同数量，默认约 80% 和 90%。分数相同的样本使用固定随机顺序打破并列。这个比较衡量排序质量和筛选覆盖的取舍。

| 10 月 19–25 日已使用过的档案诊断样本 | 沿用当前空位 | 直接空位回归 | 变化量回归 |
| --- | ---: | ---: | ---: |
| 整体 MAE（5,057 条） | 3.728 | 3.635 | 3.615 |
| 实际只有 0–2 空位时的 MAE（498 条） | 3.813 | 4.633 | 4.542 |
| 约 80% 覆盖时误选满站（各选 4,045 条） | 38 | 44 | 44 |
| 约 90% 覆盖时误选满站（各选 4,551 条） | 72 | 67 | 70 |

真实 GBFS 每分钟采集已在本机终端会话启动，写入 `data/status.sqlite`，日志为 `data/status-collection.log`。终端采集依赖当前会话和本机联网。需要续采时可执行本 README 的持续采集命令，先确认没有另一个采集进程。积累未参与训练和选参的普通周后，复用上面的比较命令，将数据库切换为 `data/status.sqlite`，设置该周的起止时间，并加上 `--allow-live-gbfs`。档案训练与实时测试使用不同数据来源，报告应同时保留共同站点覆盖和日期。

## Logistic 二分类 · 15 分钟后有没有空位

分类器复用空位变化量回归的特征、站点范围、时间切分和真实标签匹配逻辑。直接用 `目标可用空位 >= 1` 作为标签，保留暂停还车、缺失及过期数据的状态。使用类别平衡权重，在验证集按满站类的平均精度选择正则化参数。默认推理阈值在验证集选取，目标约保留 80% 候选。分类接口输出 `predicted_has_available_dock` 布尔值，无法预测时输出 null。

```bash
python dock_classifier.py train --db data/citybikes-202609-10.sqlite \
  --reference-regression artifacts/citybikes-dock-change-15.joblib \
  --model artifacts/citybikes-logistic-15.joblib > artifacts/citybikes-logistic-15.json
python dock_classifier.py evaluate --model artifacts/citybikes-logistic-15.joblib
python dock_classifier.py predict --db data/citybikes-202609-10.sqlite \
  --model artifacts/citybikes-logistic-15.joblib \
  --station-id 00802fe6-4268-4dc8-8f55-0dbbc1665ce6 --at 2026-10-19T04:04:22Z --horizon 15
```

在 `compare_return_coverage.py` 的既有命令中加上 `--classifier artifacts/citybikes-logistic-15.joblib`，可在相同站点和时间样本上比较分类器与回归。积累普通周官方快照后，将数据库换成 `data/status.sqlite`，选定日期并添加 `--allow-live-gbfs`。排行比较使用分类器分数；固定门槛的结果见分类器训练报告。

| 10 月 19–25 日档案诊断（5,057 条） | 沿用当前空位 | 直接预测空位 | 预测空位变化 | Logistic |
| --- | ---: | ---: | ---: | ---: |
| 约 80% 覆盖时误推满站（各选 4,045 条） | 38 | 44 | 44 | 51 |
| 约 90% 覆盖时误推满站（各选 4,551 条） | 72 | 67 | 70 | 75 |

这段档案已有模型诊断结果，分类器目前作为单独候选保存。常规推荐未切换到分类器。

### 按时段评估

在同覆盖率比较命令中加入 `--by-time-of-day`，会按波士顿当地的预测基准时间输出 24 小时样本数，以及凌晨、早高峰、白天、晚高峰和夜间五段的结果。每段重新按该段样本数分配候选名额，各方法在段内推荐相同数量。全周汇总仍在全体样本中排序，因此分段误推次数之和可能与全周误推次数不同。无满站样本的时段没有检验满站识别能力。

### Logistic 的离散时段特征

在分类器的训练命令中加入 `--time-band`，将预测起点的波士顿当地时间分为 00:00–06:00、06:00–10:00、10:00–16:00、16:00–20:00、20:00–24:00。分类器保留原有的小时周期特征、星期几和周末标记。时段特征由预测基准时间得到，训练、分段报告和推理使用相同边界。新版预测会返回 `time_band`。

```bash
python dock_classifier.py train --db data/citybikes-202609-10.sqlite \
  --reference-regression artifacts/citybikes-dock-change-15.joblib \
  --model artifacts/citybikes-logistic-time-band-15.joblib --time-band \
  > artifacts/citybikes-logistic-time-band-15.json
```

在比较命令中将 `--classifier` 指向这个文件，加入 `--by-time-of-day` 可复核时段效果。新版作为候选单独保存，比较沿用 10 月 19–25 日的档案诊断集。

### 低空位及连续下降规则验证

`evaluate_return_guards.py` 仅做离线同覆盖率对照。它用分类器原有验证期，比较把当前空位至多为 2、3 或 5 的候选移到 Logistic 排名后面的规则，以及在按当前空位排序时遇到连续下降就扣 1、2 或 3 个空位的规则。连续下降定义为过去 5 或 10 分钟至少观察到两次下降且无反弹。历史不足、过期或中间有长缺测时，趋势记为未知。两组规则分别按验证集误推满站最少选参数，再固定参数测试。四种方法都推荐同样多的候选（约 80%）。

```bash
python evaluate_return_guards.py --classifier artifacts/citybikes-logistic-time-band-15.joblib \
  --validation-db data/citybikes-202609-10.sqlite \
  --test-db data/citybikes-202609-10.sqlite \
  --start 2026-10-19T00:00:00-04:00 --end 2026-10-26T00:00:00-04:00 \
  > artifacts/citybikes-return-guards-archive.json
```

普通周独立测试使用 `--test-db data/status.sqlite --allow-live-gbfs` 和该周起止时间，验证库仍为 10 月档案。程序报告按起始空位数量分层的误推，并单列 16:00–20:00 的连续下降样本数和之后满站次数。短实时窗口只用于验证数据和流程。

10 月验证集选出 3 桩低空位降权，以及 5 分钟连续下降时扣 3 桩。10 月 19–25 日档案诊断中，按当前空位、原 Logistic、低空位降权、连续下降扣分依次误推 38、52、43、41 次。11 月 25 日感恩节前夕 15:35–17:35 的官方 GBFS 试跑中，四种方法依次误推 16、1,011、43、19 次。实时试跑使用 27,445 条候选样本、259 个共同站点，其中 1,857 条标签为实际满站。分钟级预测窗口相互重叠，且这段数据来自节日前夕，普通周结果仍待验证。

### 概率校准离线试验

通过 `calibrate_dock_probability.py` 在 10 月 26–28 日快照上校准带时段的 Logistic 分数，用 10 月 29–30 日档案和 11 月 25 日 15:35–17:35 的官方 GBFS 快照检验。报告比较整体历史比例、按当前空位分组的基线、校准后的 Logistic，并单列晚高峰与当前 0–2 个空位的结果。复现命令见[英文说明](README.md#offline-probability-calibration)。

档案测试上两种按输入给出概率的方法的 Brier 分数均约为 0.0330。实时试跑中 Logistic 的 Brier 分数为 0.0671，按当前空位分组的基线为 0.0549。实时样本中，当前只有 0–2 个空位时，校准后的平均预测有位概率为 95.6%，实际有位比例为 61.4%。此结果仅用于离线分析，预测接口仍输出布尔判断。完整普通周概率验证待进行。

### 春季概率试验 · 2026 年 5 月和 6 月测试

用 3 月快照训练和选模型，4 月 1–7 日做分类器初步检查，4 月 8–30 日校准。5 月和 6 月用来独立测试概率。实验不使用由 7–8 月行程生成的流量组。6 月文件按 UTC 截止，因此测试不含波士顿当地时间 6 月 30 日的最后四小时。数据来源和复现命令见[英文说明](README.md#spring-probability-experiment--may-and-june-2026-tests)。

5 月和 6 月分别有 8,868 和 11,011 条配对测试记录。校准 Logistic 的 Brier 分数分别为 0.0268 和 0.0365，按当前空位分组的基线分别为 0.0261 和 0.0355。两月合并，当前只有 0–2 个空位时，Logistic 平均预测 95.5% 有位，实际为 88.0%，空位分组基线平均为 90.6%。此次结果仅用于离线评估。

## 仓库内容与原始数据下载

仓库包含代码、Notebook、评估报告、PPT、Word 和已训练的模型。原始行程及真实站点数据库保存在本地，可按照上文的采集和导入命令重建。请从官方下载以下行程文件，将 ZIP 解压后按表中的名称保存。

| 本地文件 | 官方下载 |
| --- | --- |
| `2607.csv` | [2026 年 7 月行程](https://s3.amazonaws.com/hubway-data/202607-bluebikes-tripdata.zip) |
| `2608.csv` | [2026 年 8 月行程](https://s3.amazonaws.com/hubway-data/202608-bluebikes-tripdata.zip) |
| `2609.csv` | [2026 年 9 月行程](https://s3.amazonaws.com/hubway-data/202609-bluebikes-tripdata.csv) |

各实验所用 CityBikes 月度快照的链接见上文。原始数据适用发布方条款。 [Bluebikes 数据许可](https://bluebikes.com/data-license-agreement) · [CityBikes 数据条款](https://data.citybik.es/#terms)
