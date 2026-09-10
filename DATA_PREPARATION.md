# 文本資料準備與盤點

本工具只檢查已提供的 CSV，不下載或載入模型、不連網抓資料、不清理或改寫原文，
不排序、不去重、不聚類，也不真正切塊。現有 FinGPT 多維分析功能維持原樣。

## 預期資料位置

預設位置取自 config.py，相對路徑以專案目錄為基準：

~~~text
data/raw/news/NVDA_news_raw.csv
data/raw/earnings_call/NVDA_earnings_call_raw.csv
data/raw/earnings_call/NVDA_earnings_call_manual.csv
data/raw/ten_k/NVDA_10k_raw.csv
~~~

前三種主要來源 News、原始 EC、10-K 為必要檔案。
manual EC 視為選用補充檔：缺少時記錄 missing / warning，strict 不因此失敗；
若檔案已提供，其必要欄位或日期錯誤仍會造成 strict 失敗。
這兩份 EC 分別盤點，不自動合併，也不推定兩者是否重複。

## 第一條指令

~~~bash
conda activate fingpt4090
cd /path/to/us-stock-transformer-4090
python inspect_text_datasets.py
~~~

一般模式：即使缺少檔案或 CSV 無法讀取，仍為四個來源產生完整狀態報告。
成功寫完報告回傳 0；報告寫入失敗則非零。

~~~bash
python inspect_text_datasets.py --strict
~~~

strict 模式遇到必要檔案缺失、CSV 讀取失敗、必要欄位缺失、空白或不可解析 date，
會在產生報告後回傳非零。僅缺建議欄位、缺 manual EC 或長文警告不會使 strict 失敗。
只有表頭的空 CSV 是 warning；連表頭都沒有的零位元組 CSV 是 error。
非空原文或標題中的空白值會明確警告；結構檢查通過不代表文本已適合正式模型分析。

使用測試或其他路徑：

~~~bash
python inspect_text_datasets.py \
  --news-file /path/news.csv \
  --earnings-call-file /path/ec.csv \
  --manual-earnings-call-file /path/manual_ec.csv \
  --ten-k-file /path/tenk.csv \
  --output-dir /path/inspection_reports \
  --strict
~~~

CLI 指定的相對路徑以目前工作目錄為基準。省略的檔案參數仍使用 config.py。
工具拒絕讓報告路徑指向任何輸入檔案，包括符號連結／硬連結，避免覆寫原始 CSV。

## 報告內容與定義

預設在專案根目錄產生以下報告；可用 --output-dir 分開保存每次盤點：

- dataset_inspection_report.json：完整統計、每日／每週筆數，以及 EC／10-K 每筆正文的字元長度。
- dataset_inspection_report.md：可閱讀的來源摘要與統計。
- dataset_inspection_report.html：內嵌 CSS、UTF-8 單檔離線表格；綠／黃／紅表示 OK／warning／error。

固定輸出名稱會在重跑時更新；三種報告名稱均已加入 .gitignore。
所有報告都只保存路徑、欄位名稱、統計與錯誤分類，不包含原文、URL 欄位值、
無法解析的日期樣本、token 或環境變數。CSV 讀取錯誤不原樣輸出可能夾帶資料的例外訊息。

檔案 SHA-256 以 1 MiB 區塊串流計算；CSV 統計目前使用 pandas 在記憶體分析，
因此 SHA 計算為串流不代表整個 CSV 分析是固定記憶體用量。

統計規則：

- 空值：CSV 空儲存格。另計包含空格／換行的空白比例。
- NA、null 等非空字串保留原樣，不自動當作空值。
- 比例使用 0 到 1；空資料的比例與分布為 null，不假裝成 0。
- 完全重複：所有欄位相同；另計來源指定的 key 重複，均不包含第一次出現。
- 缺少重複檢查需要的欄位時，該統計為 null，不回報假造的零。
- date 以 mixed 格式解析並轉成 UTC 進行統計；模糊數字日期採月在日前。
  真正上線前應統一 ISO 日期／時區，尤其避免 01/02/2024 之類歧義。
- 日期解析成功不等於公開日期正確。原始 date 不會被修改。
- 每日／每週統計僅包含實際有資料的日期／週，週一為週起點，缺日缺週不補零。

News 的分析文本逐列依序選擇 title + content、title + summary、title。
空白 content 才退回 summary。報告另外統計標題空白、正文空白、摘要空白、
只有標題、有標題加正文／摘要，以及兩種重複 key。

EC／10-K 使用 title + content，沒有 title 時使用 content。
報告提供每筆原始 content 字元長度及合成分析文本的長度分布：
minimum、median、p90、p95、p99、maximum。
文字長度分布包含空白分析文本的 0 長度，並另外明列空白筆數。

超過 4,000 字元：建議檢查；超過 8,000 字元：高度可能需要切塊。
這只是字元層級的 provisional warning，不能換算成可靠 token 數。
正式切塊應使用取得權限後的實際 tokenizer。本工具不匯入 Llama tokenizer。

10-K 的 Item 1、1A、7、7A 檢查會分別統計 content 出現與 section 標記。
提到 Item 不代表已抽出完整章節，也不能據此斷言缺少整份文件的某章。

## News 後續流程（本次不實作）

title + content／summary
→ 基本文字清理
→ 完全去重
→ Sentence Transformer embedding
→ BERTopic 或相似度聚類
→ 群組品質評估
→ 代表新聞
→ FinGPT 多維分析

未來獨立新聞預處理模組預計產生：

- news_id
- cluster_id
- cluster_size
- cluster_similarity
- cluster_start_date
- cluster_end_date
- cluster_span_days
- is_representative
- representative_text

cluster_similarity 預計採群內平均 pairwise cosine similarity；
代表新聞選擇最接近群中心者。cluster_size、similarity、span 都由程式計算，
不可要求 LLM 猜測。未來需明確定義 singleton 群的 similarity，以及重複報導的計數方式。

FinGPT 分析 representative_text，聚類 metadata 可透過未來選用 context 介面傳入。
客觀欄位保留在外層資料中，不改成模型猜測欄位。
聚類模組必須與 fingpt_multidimensional.py 分離，本工具不新增任何聚類依賴或程式。

## Earnings Call 後續流程（本次不實作）

講者／章節辨識
→ Prepared Remarks 與 Q&A 分段
→ tokenizer-based chunking
→ FinGPT 多維分析
→ 整場彙整

先辨認正文究竟是 transcript、CFO commentary 或新聞稿，不能只因檔名為 EC
就視為完整逐字稿。盤點會顯示 speaker、section、chunk_id 是否存在。

## 10-K 後續流程（本次不實作）

Item 1、1A、7、7A 章節抽取
→ tokenizer-based chunking
→ FinGPT 多維分析
→ 文件彙整

章節抽取前需檢查 XBRL、HTML 雜訊與原文截斷，不能把字元數固定的片段當成完整申報文件。

## 防止未來資訊洩漏

預測截止時間 t 的任何特徵，只能使用 t 當下或以前已公開的內容。

- 10-K 使用 filing_date／實際公開時間，不可直接使用 fiscal year end。
- EC 使用實際公開／舉行時間，不可把財報年度或季末當成發表日期。
- 新聞聚類、中心、代表新聞與所有群組統計只能使用截至 t 的新聞。
- 不可先以全期間新聞分群，再把最終 cluster_size 或代表新聞回填到過去。
- 後续整合應保存計算截止時間與來源識別碼，以便重建當時可得資訊。

目前只建立資料盤點與後續規格，沒有實作上述聚類、章節抽取或切塊。

## 離線測試

~~~bash
python -m py_compile inspect_text_datasets.py test_inspect_text_datasets.py
python -m unittest -v
python inspect_text_datasets.py
git diff --check
~~~

新增測試全部使用 tempfile 與 synthetic CSV；不以預期存在的真實資料為測試前提。
既有完整測試包含 mock FinGPT 及小型 synthetic CPU Transformer 測試，均不載入預訓練模型。
