# Requirements Document

## Introduction

本文件定義了 **kiro-FIFA-Predictor** Kiro Power 的功能需求。此 Power 提供 2026 年 FIFA 世界盃賽事預測功能，透過 MCP 工具整合至 Kiro 聊天介面，允許使用者查詢單場比賽預測、小組賽晉級預測、冠軍預測，並在賽後自動更新數據以提升預測準確度。系統採用 Dixon-Coles Poisson、Elo 評分與歷史對戰紀錄的集成模型，支援三種教練風格分析。

## Glossary

- **Prediction_Engine**: 預測引擎，整合多種統計模型（Poisson、Elo、歷史對戰、動態因子）的集成系統，負責計算比賽結果機率
- **Dixon_Coles_Poisson_Model**: 基於 Poisson 分佈的進球預測模型，使用 Dixon-Coles 低分修正，計算公式為 attack_strength × defense_weakness × neutral_factor
- **Elo_Model**: 基於 Elo 評分系統的勝率預測模型，中立場地無主場加成，但主辦國（美國/加拿大/墨西哥）在其國內比賽時獲得 +50 分加成
- **Ensemble_Model**: 集成模型，將多個子模型按權重組合（Poisson 0.40、Elo 0.25、歷史對戰 0.15、動態因子 0.20）
- **Dynamic_Factor**: 動態因子，包含連勝加成（±5%）、首場罰分、疲勞效應（休息不足 3 天 -3%）、復仇因子（上屆被淘汰 +3%）
- **Coach_Style**: 教練風格，系統提供三種分析視角：分析師（統計導向）、反向思考者（弱隊偏好）、戰術家（戰術因素導向）
- **Recalibration_Process**: 重新校準流程，在每場比賽結果更新後自動調整模型權重與球隊動態數據
- **Confidence_Index**: 信心指數，表示預測結果可靠程度的數值指標
- **MCP_Tool**: Model Context Protocol 工具，Kiro Power 中提供特定功能的可呼叫端點
- **Team_Profile**: 球隊資料檔，包含 FIFA 排名、Elo 評分、近 10 場統計、中立場地勝率等完整數據
- **Accuracy_Tracker**: 準確度追蹤器，記錄並計算預測命中率與誤差指標的子系統
- **Over_Under_Prediction**: 大小球預測，預測比賽總進球數是否超過或低於特定門檻值

## Requirements

### 需求 1：單場比賽預測

**使用者故事：** 作為使用者，我想要預測兩支球隊的比賽結果，以便在觀賽前了解可能的比分與勝負機率。

#### 驗收標準

1. WHEN 使用者提供兩支球隊名稱請求比賽預測, THE Prediction_Engine SHALL 回傳最可能的前 3 個比分及其對應機率，每個機率值以百分比表示並精確至小數點後 1 位
2. WHEN 使用者提供兩支球隊名稱請求比賽預測, THE Prediction_Engine SHALL 回傳勝/平/負三方機率，每個機率值以百分比表示並精確至小數點後 1 位，三者總和為 100.0%
3. WHEN 使用者提供兩支球隊名稱請求比賽預測, THE Prediction_Engine SHALL 回傳一個介於 0 至 100 之間的整數 Confidence_Index
4. WHEN 使用者提供兩支球隊名稱請求比賽預測, THE Prediction_Engine SHALL 回傳大小球預測，包含以 2.5 球為門檻的「超過」與「低於」機率，兩者各精確至小數點後 1 位且總和為 100.0%
5. IF 使用者提供的球隊名稱不存在於 48 支參賽隊伍中, THEN THE Prediction_Engine SHALL 回傳錯誤訊息並列出最多 3 個最相似的球隊名稱建議
6. WHEN 使用者指定 Coach_Style（分析師、反向思考者、戰術家其中之一）, THE Prediction_Engine SHALL 根據指定風格調整預測權重並於回傳結果中標註所使用的風格名稱
7. IF 使用者指定的 Coach_Style 不屬於三種有效風格（分析師、反向思考者、戰術家）, THEN THE Prediction_Engine SHALL 回傳錯誤訊息並列出三種有效的 Coach_Style 選項
8. THE Prediction_Engine SHALL 支援以英文名稱或常用中文名稱辨識 48 支參賽隊伍

### 需求 2：小組賽預測

**使用者故事：** 作為使用者，我想要預測某個小組的最終排名，以便了解哪些球隊可能晉級淘汰賽。

#### 驗收標準

1. WHEN 使用者指定一個小組（A 至 L）, THE Prediction_Engine SHALL 使用 Ensemble_Model 模擬該小組全部 6 場循環賽，並依據積分（高者優先）、淨勝球（高者優先）、進球數（高者優先）之順序排列，回傳該小組 4 支球隊的預測最終排名（第 1 名至第 4 名）
2. WHEN 使用者指定一個小組, THE Prediction_Engine SHALL 為每支球隊顯示：比賽場數、勝場數、平場數、負場數、預測積分、進球數、失球數與淨勝球
3. WHEN 使用者指定一個小組, THE Prediction_Engine SHALL 標示預測晉級的球隊，其中前兩名標示為「確定晉級」，第三名標示其晉級機率（以百分比表示，介於 0% 至 100%），機率達 50% 以上者額外標示為「可能晉級」
4. IF 使用者指定的小組代號不存在（非 A 至 L）, THEN THE Prediction_Engine SHALL 回傳錯誤訊息並列出所有有效的小組代號（A 至 L）
5. WHEN 使用者指定一個小組, THE Prediction_Engine SHALL 回傳該小組 6 場比賽各自的預測比分，以呈現排名推導依據

### 需求 3：冠軍預測

**使用者故事：** 作為使用者，我想要預測世界盃冠軍，以便了解各隊奪冠機率與可能的淘汰賽路線。

#### 驗收標準

1. WHEN 使用者請求冠軍預測, THE Prediction_Engine SHALL 使用至少 10,000 次蒙地卡羅模擬計算淘汰賽完整賽程（32 強賽、16 強賽、8 強賽、4 強賽、季軍戰、決賽），並回傳預測冠軍球隊與該隊奪冠機率
2. WHEN 使用者請求冠軍預測, THE Prediction_Engine SHALL 為 32 支進入淘汰賽的球隊各自顯示晉級至每一輪次（16 強、8 強、4 強、決賽、奪冠）的機率，機率以百分比呈現並精確至小數點後一位
3. WHEN 使用者請求冠軍預測, THE Prediction_Engine SHALL 顯示前 5 名最可能奪冠的球隊及其奪冠機率，依機率由高至低排序
4. IF 小組賽尚未全部完成且無法確定 32 支淘汰賽球隊, THEN THE Prediction_Engine SHALL 先基於小組賽預測結果推算晉級球隊，再執行淘汰賽模擬，並標註結果包含小組賽預測假設
5. WHEN 使用者請求冠軍預測, THE Prediction_Engine SHALL 回傳一個介於 0 至 100 之間的 Confidence_Index，反映本次模擬結果的收斂程度

### 需求 4：賽後數據更新與模型重新校準

**使用者故事：** 作為使用者，我想要在比賽結束後更新實際結果，以便系統能根據真實數據改善預測準確度。

#### 驗收標準

1. WHEN 使用者觸發結果更新指令, THE Recalibration_Process SHALL 在 30 秒內從外部數據源取得最新比賽結果
2. IF 外部數據源在 30 秒內無法回應, THEN THE Recalibration_Process SHALL 提示使用者手動輸入比賽結果作為備援方式
3. WHEN 新比賽結果被取得, THE Recalibration_Process SHALL 比較預測結果與實際結果，並記錄精確比分差異、勝負方向是否正確、及進球數誤差
4. WHEN 新比賽結果被取得, THE Recalibration_Process SHALL 自動調整 Ensemble_Model 的權重參數，且調整後每個子模型權重須維持在 0.10 至 0.60 之間，所有權重總和為 1.00
5. WHEN 新比賽結果被取得, THE Recalibration_Process SHALL 更新相關球隊的 Dynamic_Factor 數據，包含連勝/連敗記錄、疲勞狀態、及復仇因子
6. WHEN 重新校準完成, THE Recalibration_Process SHALL 回報各子模型權重調整前後數值、精確比分命中率變化、及勝負方向命中率變化
7. IF 累計至少 5 場比賽結果後精確比分命中率低於 5%, THEN THE Recalibration_Process SHALL 調整 Ensemble_Model 權重，單次調整幅度不超過各權重值的 ±0.05
8. IF 累計至少 5 場比賽結果後勝負方向命中率低於 50%, THEN THE Recalibration_Process SHALL 產出跨聯盟比賽系統性偏差分析報告，列出各聯盟對戰的預測命中率與偏差方向

### 需求 5：預測準確度追蹤

**使用者故事：** 作為使用者，我想要查看系統的預測準確度，以便評估預測結果的可靠性。

#### 驗收標準

1. WHEN 使用者查詢準確度, THE Accuracy_Tracker SHALL 顯示精確比分命中率（預測比分完全正確的場次除以已完賽且已更新結果的總場次，以百分比呈現，精確至小數點後一位）
2. WHEN 使用者查詢準確度, THE Accuracy_Tracker SHALL 顯示勝負方向命中率（正確預測勝/平/負的場次除以已完賽且已更新結果的總場次，以百分比呈現，精確至小數點後一位）
3. WHEN 使用者查詢準確度, THE Accuracy_Tracker SHALL 顯示平均進球誤差（每場比賽中預測總進球數與實際總進球數之差的絕對值，加總後除以已完賽場次數，精確至小數點後兩位）
4. WHEN 使用者查詢準確度, THE Accuracy_Tracker SHALL 顯示各 Coach_Style（分析師、反向思考者、戰術家）的個別精確比分命中率、勝負方向命中率與平均進球誤差
5. WHEN 使用者查詢準確度, THE Accuracy_Tracker SHALL 顯示信心校準分析，將預測依 Confidence_Index 分為三個區間（0-33 低信心、34-66 中信心、67-100 高信心），並分別顯示各區間的勝負方向命中率
6. WHEN 使用者查詢準確度, THE Accuracy_Tracker SHALL 顯示跨聯盟對戰準確度分析，依對戰聯盟組合（如 UEFA 對 CONMEBOL）分組顯示勝負方向命中率
7. IF 已更新結果的比賽場次少於 3 場, THEN THE Accuracy_Tracker SHALL 回傳訊息說明目前樣本數不足（顯示目前已完成場次數），無法提供具統計意義的準確度數據
8. IF 尚未有任何比賽結果更新, THEN THE Accuracy_Tracker SHALL 回傳訊息說明目前無已完賽數據，無法提供準確度數據

### 需求 6：球隊資料查詢

**使用者故事：** 作為使用者，我想要查詢特定球隊的詳細數據，以便深入了解該隊的實力指標。

#### 驗收標準

1. WHEN 使用者查詢特定球隊資料, THE MCP_Tool SHALL 支援模糊比對（包含部分名稱、常見縮寫、中文隊名），並以表格格式顯示該隊的 Team_Profile
2. THE Team_Profile SHALL 包含以下數據欄位：FIFA 排名、FIFA 積分、Elo 評分、近 10 場平均進球數（小數點後兩位）、近 10 場平均失球數（小數點後兩位）、近 10 場勝率（百分比）、近 10 場平率（百分比）、近 10 場負率（百分比）、中立場地勝率（百分比）、世界盃歷史最佳成績、對前 20 名球隊勝率（百分比）、目前連勝/連敗記錄、所屬聯盟、上半場進球佔比（百分比）、下半場進球佔比（百分比）、十二碼大戰勝率（百分比）、世界盃首場比賽勝率（百分比）、零封率（百分比）、未進球率（百分比）
3. IF 使用者查詢的球隊名稱無法匹配 48 支參賽隊伍中的任何一支, THEN THE MCP_Tool SHALL 回傳錯誤訊息指出查無此隊，並列出最多 3 支名稱最相似的球隊作為建議
4. IF 使用者查詢的球隊名稱同時模糊匹配到多支球隊, THEN THE MCP_Tool SHALL 列出所有匹配結果供使用者選擇，而非直接顯示任一隊的 Team_Profile

### 需求 7：集成模型計算

**使用者故事：** 作為開發者，我想要系統使用多模型集成方式計算預測，以便提供更準確且穩健的預測結果。

#### 驗收標準

1. THE Dixon_Coles_Poisson_Model SHALL 使用 attack_strength = (team_goals_avg / league_avg) × confederation_coeff 與 defense_weakness = (opponent_conceded_avg / league_avg) 計算預期進球數 lambda = attack_strength × defense_weakness × neutral_factor，並針對 0-0、1-0、0-1、1-1 四種低分比分套用 Dixon-Coles tau 修正參數，最終輸出 5×5 比分機率矩陣
2. THE Elo_Model SHALL 使用公式 P(A) = 1 / (1 + 10^((Elo_B - Elo_A + home_advantage) / 400)) 計算勝率，中立場地比賽中 home_advantage 設為 0
3. WHEN 主辦國（美國、加拿大或墨西哥）在其本國場地進行比賽, THE Elo_Model SHALL 給予該主辦國 +50 分 Elo 加成
4. THE Ensemble_Model SHALL 以下列預設權重組合各模型輸出：Poisson 0.40、Elo 0.25、歷史對戰 0.15、動態因子 0.20，所有權重總和為 1.00，且每個權重須維持在 0.10 至 0.60 之間
5. WHEN 球隊連勝場次 ≥ 3 場, THE Dynamic_Factor SHALL 對該隊預測勝率施加 +5% 加成；WHEN 球隊連敗場次 ≥ 3 場, THE Dynamic_Factor SHALL 對該隊預測勝率施加 -5% 減損
6. WHEN 球隊的上一場比賽與當前比賽間隔不足 3 天, THE Dynamic_Factor SHALL 對該隊預測勝率施加 -3% 疲勞效應調整
7. WHEN 球隊面對上屆世界盃（2022 年卡達世界盃）將其淘汰的對手, THE Dynamic_Factor SHALL 對該隊預測勝率施加 +3% 復仇因子調整
8. IF 任一子模型計算失敗, THEN THE Ensemble_Model SHALL 以剩餘正常運作的子模型重新分配權重（按原比例重分）並標註缺失的模型

### 需求 8：教練風格系統

**使用者故事：** 作為使用者，我想要從不同分析視角查看預測，以便獲得更全面的比賽分析。

#### 驗收標準

1. WHEN 使用者選擇「分析師」風格, THE Coach_Style SHALL 直接輸出統計模型計算的勝率與比分預測，不對機率值進行任何加成或調整
2. WHEN 使用者選擇「反向思考者」風格, IF FIFA 排名較低球隊的模型預測勝率低於 35%, THEN THE Coach_Style SHALL 將該球隊勝率提升至 35%–40% 範圍，並推薦冷門比分作為建議比分
3. WHEN 使用者選擇「戰術家」風格, THE Coach_Style SHALL 檢查近期連勝/連敗趨勢、歷史交鋒復仇因素、及賽程密度疲勞狀態，並據此調整預測機率
4. THE MCP_Tool SHALL 在預測結果中同時顯示三種 Coach_Style 的預測，每種風格包含：調整後勝率、建議比分、及該風格的敘事文字說明
5. WHEN 使用者未指定 Coach_Style, THE MCP_Tool SHALL 預設使用「分析師」風格作為主要預測結果
6. WHEN 使用者輸入風格關鍵字（「aggressive」或「激進」對應反向思考者、「conservative」或「保守」對應分析師、「balanced」或「平衡」對應戰術家）, THE MCP_Tool SHALL 啟用對應的 Coach_Style 進行預測
7. THE Coach_Style SHALL 為每種風格產生固定開頭的敘事文字：分析師使用「根據統計分析…」、反向思考者使用「從冷門角度…」、戰術家使用「考量戰術因素…」

### 需求 9：數據管理與初始化

**使用者故事：** 作為使用者，我想要系統能正確載入與管理球隊數據，以便預測基於完整且最新的資訊。

#### 驗收標準

1. WHEN MCP_Tool 啟動時, THE MCP_Tool SHALL 從本地 data/teams.json 載入 48 支 2026 世界盃參賽隊伍的 Team_Profile 數據，並驗證恰好存在 48 筆隊伍記錄且每筆記錄無缺漏必填欄位
2. WHEN MCP_Tool 啟動時, THE MCP_Tool SHALL 從本地 data/groups.json 載入 12 個小組（A 至 L）的分組資訊，並驗證每組恰好包含 4 支隊伍，同時從 data/schedule.json 載入完整賽程表
3. IF 啟動時數據完整性驗證失敗（隊伍數量不等於 48、任一小組隊伍數量不等於 4、或存在缺漏必填欄位）, THEN THE MCP_Tool SHALL 中止啟動流程並回傳錯誤訊息指出驗證失敗的具體項目
4. IF 外部數據源於 30 秒內無法建立連線, THEN THE MCP_Tool SHALL 切換至本地備援數據（由 scripts/fallback_data.py 產生之靜態快照）繼續提供預測服務，並於回應中標示數據來源為備援資料
5. WHEN 比賽結果更新完成, THE MCP_Tool SHALL 將更新後的數據寫入 data/match_results.json，寫入過程中先寫入暫存檔再以重新命名方式取代原檔，避免寫入中斷導致資料損毀
6. WHEN 一次預測完成, THE MCP_Tool SHALL 將該次預測紀錄附加至 data/predictions_log.json，紀錄內容須包含預測時間戳記、預測對象比賽識別碼、預測結果、以及所使用的模型參數摘要

### 需求 10：輸出格式與顯示

**使用者故事：** 作為使用者，我想要預測結果以清晰美觀的格式呈現，以便快速理解預測內容。

#### 驗收標準

1. WHEN 顯示單場預測結果, THE MCP_Tool SHALL 呈現包含以下欄位的結構化區塊：雙方球隊名稱與國旗 emoji、預測比分、機率最高的前 3 組比分與其百分比、勝/平/負機率、Confidence_Index 百分比、預期進球數（xG）、以及大小球（Over/Under 2.5）預測
2. WHEN 顯示小組排名, THE MCP_Tool SHALL 以表格格式呈現各隊資料，欄位依序為：排名（Rank）、球隊名（Team）、積分（P）、勝場（W）、平場（D）、負場（L）、進球（GF）、失球（GA）、淨勝球（GD），並依積分由高至低排序
3. WHEN 顯示冠軍預測, THE MCP_Tool SHALL 以淘汰賽對陣樹狀結構呈現各輪次（十六強、八強、四強、決賽）的對陣組合與預測晉級球隊
4. WHEN 顯示球隊資料, THE MCP_Tool SHALL 以分類區塊格式呈現 Team_Profile，分類包含：基本資訊（Basic Info）、近期戰績（Recent Form）、世界盃歷史（World Cup History）、進階數據（Advanced Stats）
5. THE MCP_Tool SHALL 在所有預測輸出結果的末尾標註 data_updated_at 時間戳記（ISO 8601 格式）與 model_version 版本字串
6. IF 輸出環境為 CLI 終端機, THEN THE MCP_Tool SHALL 使用 Rich 函式庫渲染輸出，包含框線、色彩與對齊排版
7. IF 輸出環境為 Kiro 聊天介面, THEN THE MCP_Tool SHALL 以結構化 Markdown 格式輸出，確保在聊天介面中正確渲染
