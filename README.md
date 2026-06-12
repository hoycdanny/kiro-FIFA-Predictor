# 🏆 kiro-FIFA-Predictor

2026 FIFA 世界盃比賽預測 Kiro Power — 透過 MCP 工具在 Kiro 聊天介面中進行比賽預測分析。

## 功能特色

- **單場比賽預測** — 勝/平/負機率、前 3 名最可能比分、信心指數、大小球預測
- **小組賽模擬** — 完整循環賽模擬、積分表排名、晉級預測
- **冠軍預測** — 蒙地卡羅模擬（10,000+ 次）完整淘汰賽路線
- **賽後校準** — 輸入真實結果後自動調整模型權重
- **準確度追蹤** — 命中率、方向正確率、進球誤差統計
- **球隊資料查詢** — 48 支參賽隊伍完整數據檔案
- **三種教練風格** — 分析師（統計導向）、反向思考者（冷門偏好）、戰術家（戰術因素）
- **多語言隊名** — 支援英文、中文、縮寫的模糊比對

## 系統需求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (建議) 或 pip

## 安裝

```bash
# 使用 uv (推薦)
uv sync

# 或使用 pip
pip install -e .

# 安裝開發依賴 (含測試工具)
pip install -e ".[dev]"
```

## 在 Kiro 中使用 (作為 Power)

本專案已配置為 Kiro Power，安裝後可直接在 Kiro 聊天中使用。

### 自動載入 (已配置)

Power 配置位於 `.kiro/powers/fifa-predictor/`，在此工作區中 Kiro 會自動偵測並載入。

### 手動配置 MCP

如果需要在其他工作區中使用，在 `.kiro/settings/mcp.json` 中加入：

```json
{
  "mcpServers": {
    "fifa-predictor": {
      "command": "python",
      "args": ["src/server.py"],
      "cwd": "C:\\Users\\Danny\\Desktop\\kiro-FIFA-Predictor"
    }
  }
}
```

> 請將 `cwd` 替換為你的實際專案路徑。

### 使用方式

在 Kiro 聊天中直接對話即可，Power 啟動後提供 6 個 MCP 工具：

#### 預測單場比賽

```
幫我預測巴西 vs 阿根廷
```

```
predict_match(team_a="Brazil", team_b="Germany", coach_style="戰術家")
```

#### 預測小組排名

```
預測 A 組的最終排名
```

```
predict_group(group_id="B")
```

#### 預測冠軍

```
誰最可能拿冠軍？
```

```
predict_champion(simulations=10000)
```

#### 更新比賽結果

```
update_results(manual_result="Brazil 2 - 1 Argentina")
```

#### 查看準確度

```
系統準確度如何？
```

```
accuracy_stats()
```

#### 查詢球隊資料

```
查一下法國隊的數據
```

```
team_info(team_name="France")
```

## 直接執行 MCP Server

如果想在 Kiro 之外以 CLI 方式測試：

```bash
python src/server.py
```

Server 使用 stdio 傳輸協定，啟動後會驗證資料完整性（48 隊、12 組 × 4 隊），驗證通過後開始接受請求。

## 專案結構

```
kiro-FIFA-Predictor/
├── src/
│   ├── server.py              # MCP Server 進入點
│   ├── engine/                # 預測引擎核心
│   │   ├── prediction_engine.py   # 主控引擎
│   │   ├── ensemble.py            # 集成模型
│   │   ├── dixon_coles.py         # Dixon-Coles Poisson 模型
│   │   ├── elo_model.py           # Elo 評分模型
│   │   ├── h2h_model.py           # 歷史對戰模型
│   │   ├── dynamic_factor.py      # 動態因子模型
│   │   ├── monte_carlo.py         # Monte Carlo 模擬器
│   │   └── coach_style.py         # 教練風格系統
│   ├── data/                  # 資料管理層
│   │   └── data_manager.py        # 資料載入/寫入/驗證
│   ├── tools/                 # MCP 工具實作
│   │   ├── predict_match.py
│   │   ├── predict_group.py
│   │   ├── predict_champion.py
│   │   ├── update_results.py
│   │   ├── accuracy_stats.py
│   │   └── team_info.py
│   ├── output/                # 輸出格式化
│   │   ├── formatter.py
│   │   ├── markdown_renderer.py   # Kiro 聊天用
│   │   └── rich_renderer.py       # CLI 用
│   └── utils/                 # 工具模組
│       ├── constants.py           # 48 隊名稱/常數
│       ├── team_matcher.py        # 模糊名稱比對
│       ├── validator.py           # 輸入驗證
│       └── accuracy_tracker.py    # 準確度追蹤
├── data/                      # 資料檔案
│   ├── teams.json                 # 48 隊完整資料
│   ├── groups.json                # 12 組分組
│   ├── schedule.json              # 賽程表
│   ├── match_results.json         # 比賽結果 (動態更新)
│   ├── predictions_log.json       # 預測紀錄
│   └── calibration.json           # 模型權重
├── tests/                     # 測試
│   ├── properties/                # 性質測試 (Hypothesis)
│   ├── integration/               # 整合測試
│   └── unit/                      # 單元測試
├── scripts/
│   └── fallback_data.py           # 備援資料產生器
└── pyproject.toml
```

## 預測模型說明

系統使用四個子模型的加權集成：

| 模型 | 權重 | 說明 |
|------|------|------|
| Dixon-Coles Poisson | 40% | 基於進攻/防守強度的進球分佈模型 |
| Elo 評分 | 25% | 基於歷史表現的勝率計算 |
| 歷史對戰 | 15% | 兩隊歷史交鋒紀錄 |
| 動態因子 | 20% | 連勝/連敗、疲勞、復仇等即時因素 |

權重會在賽後校準時自動調整（每次 ±0.05 以內，每項維持在 0.10–0.60 範圍）。

## 教練風格

| 風格 | 關鍵字 | 特色 |
|------|--------|------|
| 分析師 | conservative / 保守 | 純統計模型輸出，不做調整 |
| 反向思考者 | aggressive / 激進 | 偏好冷門，提升弱隊勝率至 35-40% |
| 戰術家 | balanced / 平衡 | 考量連勝/疲勞/復仇等戰術因素 |

## 測試

```bash
# 執行全部測試
pytest

# 執行含覆蓋率
pytest --cov=src --cov-report=term-missing

# 只跑性質測試
pytest tests/properties/

# 只跑整合測試
pytest tests/integration/
```

目前共 471 個測試，涵蓋 20 項正確性性質 (Property-Based Testing)。

## 技術棧

- **語言**: Python 3.11+
- **MCP SDK**: FastMCP (mcp[cli])
- **數值計算**: NumPy
- **CLI 輸出**: Rich
- **測試**: pytest + Hypothesis (PBT)
- **傳輸**: stdio (MCP 標準)

## 授權

MIT
