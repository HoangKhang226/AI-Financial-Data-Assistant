# Cấu trúc thư mục dự án Text-to-Pandas (R2AI2026)

```
.
├── configs/                              # Cấu hình hệ thống
│   ├── model_config.yaml                 # Cấu hình LLM (Qwen2.5-Coder-14B), Embedding (BGE-M3), Reranker
│   ├── index_config.yaml                 # Cấu hình Vector Store local và BM25 Index
│   └── prompts.yaml                      # System Prompt cho Code Generation, Self-Correction, Entity Extraction
│
├── data/                                 # Quản lý dữ liệu
│   ├── ViFinQA/                          # Dataset gốc từ HuggingFace (git clone)
│   │   ├── financial_statements/         # 1,973 file OCR .txt (100 tickers × ~10 năm × 2 loại)
│   │   │   └── TICKER/YEAR/DOC/          #   Ví dụ: AAA/2015/AAA_financial_statements_2015_consolidated/
│   │   │       └── DOC_extracted.txt     #   File text OCR chứa ===== PAGE N ===== và <table> HTML inline
│   │   ├── questions/                    # Bộ câu hỏi kiểm thử
│   │   │   └── questions.jsonl           #   1,012 câu hỏi tài chính (id + question)
│   │   └── code_stock.csv                # Mapping 100 mã CK ↔ tên công ty
│   │
│   ├── csv_warehouse/                    # OUTPUT Pipeline 1: Kho CSV chuẩn hóa (~146K file)
│   │   └── TICKER_YEAR_TYPE_pN_lM.csv   #   Ví dụ: AAA_2015_consolidated_p7_l214.csv
│   │
│   ├── metadata/                         # OUTPUT Pipeline 1: JSON Metadata cho mỗi bảng
│   │   └── TICKER_YEAR_TYPE_pN_lM.json   #   doc_id, ticker, year, report_type, line_number, unit, columns
│   │
│   └── index/                            # OUTPUT Pipeline 1: Chỉ mục lai
│       ├── dense_vectors/                #   BGE-M3 embedding vectors (FAISS/numpy)
│       ├── bm25_index/                   #   BM25Okapi serialized index
│       └── summaries.jsonl               #   Template-based summaries cho từng bảng
│
├── src/                                  # Mã nguồn chính
│   ├── pipeline_1_ingestion/             # PIPELINE 1: OCR Text → CSV + Metadata + Index
│   │   ├── page_splitter.py              # Đọc file .txt, tách ===== PAGE N =====, tracking line_number
│   │   ├── html_table_parser.py          # BeautifulSoup parse <table>, thuật toán ma trận 2D (rowspan/colspan)
│   │   ├── ocr_normalizer.py             # Sửa lỗi OCR: số âm ngoặc đơn, O→0, l→1, dấu phân cách số
│   │   ├── unit_detector.py              # Phát hiện đơn vị gốc trong BCTC ("Đơn vị: VND", "triệu đồng")
│   │   ├── csv_exporter.py               # Export DataFrame → CSV vào data/csv_warehouse/
│   │   ├── metadata_builder.py           # Sinh JSON metadata (doc_id, ticker, year, type, line_number, unit)
│   │   ├── summary_generator.py          # Template-based summary (không dùng LLM, dựa trên metadata)
│   │   ├── index_builder.py              # BGE-M3 dense embedding + BM25Okapi sparse index
│   │   └── pipeline.py                   # Điều phối Pipeline 1: run_ingestion(input_dir, output_dir)
│   │
│   ├── pipeline_2_query/                 # PIPELINE 2: Question → Answer + Submission
│   │   ├── entity_extractor.py           # Regex + code_stock.csv → ticker, year, report_type, target_unit
│   │   ├── metadata_filter.py            # Lọc cứng JSON metadata theo ticker + year + report_type
│   │   ├── hybrid_searcher.py            # BGE-M3 Dense + BM25 Sparse + RRF Fusion
│   │   ├── reranker.py                   # BAAI/bge-reranker-v2-m3 Cross-Encoder (open-source, local)
│   │   ├── prompt_builder.py             # Xây prompt XML: schema, sample rows, unit info, question
│   │   ├── code_generator.py             # Qwen2.5-Coder-14B-Instruct → sinh mã Python Pandas
│   │   ├── sandbox_executor.py           # subprocess.run() cách ly, timeout=5s, capture stderr
│   │   ├── self_correction.py            # Gửi traceback + code lỗi cho LLM sửa (max 3 retries)
│   │   ├── fallback_rag.py              # Direct RAG: trích xuất số từ DataFrame nếu code gen thất bại
│   │   ├── unit_converter.py             # Ma trận quy đổi VND↔triệu↔tỷ↔nghìn tỷ
│   │   ├── submission_formatter.py       # Đóng gói JSON theo chuẩn R2AI2026 submission format
│   │   └── pipeline.py                   # Điều phối Pipeline 2: run_query(questions, index, output)
│   │
│   └── common/                           # Module dùng chung
│       ├── llm_client.py                 # Wrapper cho Qwen2.5-Coder-14B (vLLM/transformers/llama.cpp)
│       ├── embedding_client.py           # Wrapper cho BAAI/bge-m3 (sentence-transformers)
│       ├── reranker_client.py            # Wrapper cho BAAI/bge-reranker-v2-m3 (FlagEmbedding)
│       ├── schemas.py                    # Pydantic models: Question, TableMetadata, SubmissionEntry
│       └── logger.py                     # Structured logging (JSON format)
│
├── evaluation/                           # Đánh giá chất lượng
│   ├── metrics/
│   │   ├── retrieval_metrics.py          # Precision, Recall, F2 macro cho bước retrieval
│   │   ├── execution_accuracy.py         # Tỷ lệ code chạy được + cho kết quả đúng
│   │   └── answer_accuracy.py            # So sánh answer với đáp án chuẩn (trong ngưỡng sai số)
│   └── run_eval.py                       # Chạy đánh giá End-to-End
│
├── scripts/                              # Kịch bản vận hành
│   ├── download_dataset.py               # git clone ViFinQA từ HuggingFace
│   ├── batch_ingest.py                   # Chạy Pipeline 1 trên toàn bộ 1,973 file .txt
│   ├── batch_query.py                    # Chạy Pipeline 2 trên 1,012 câu hỏi
│   └── build_submission.py               # Đóng gói submission.zip (submission.json + data/)
│
├── tests/                                # Kiểm thử tự động (pytest)
│   ├── test_html_parser.py               # Unit test: parse bảng HTML có colspan/rowspan
│   ├── test_ocr_normalizer.py            # Unit test: sửa lỗi OCR, chuẩn hóa số
│   ├── test_entity_extractor.py          # Unit test: trích xuất ticker, year, report_type
│   ├── test_unit_converter.py            # Unit test: quy đổi đơn vị VND↔triệu↔tỷ
│   └── test_end_to_end.py                # Integration test: chạy 10 câu hỏi mẫu end-to-end
│
├── requirements.txt                      # Dependencies: beautifulsoup4, pandas, sentence-transformers,
│                                         #   rank-bm25, FlagEmbedding, transformers, vllm, pydantic
└── README.md                             # Hướng dẫn cài đặt, kiến trúc, và cách chạy
```

## Thống kê dữ liệu

| Mục | Giá trị |
|---|---|
| File OCR .txt | 1,973 |
| Bảng HTML inline | 146,246 |
| Bảng có colspan | 56,941 (38.9%) |
| Bảng có rowspan | 45,338 (31.0%) |
| Công ty (tickers) | 100 |
| Giai đoạn | 2015–2025 |
| Câu hỏi kiểm thử | 1,012 |
| Dung lượng text thô | ~363 MiB |

## Ràng buộc cuộc thi R2AI2026

| Ràng buộc | Giá trị |
|---|---|
| LLM | Open-source, ≤14B params, phát hành trước 01/06/2026 |
| API đóng | ❌ Không dùng GPT-4o, Gemini, Cohere, Claude |
| Nộp bài | submission.zip = submission.json + data/*.csv |
| Giới hạn nộp | 10 bài/ngày (Public), 5 bài tổng cộng (Private) |
| Deadline | Public: 01-31/08/2026, Private: 01-03/09/2026 |

## Stack kỹ thuật

| Thành phần | Công nghệ |
|---|---|
| Code LLM | Qwen2.5-Coder-14B-Instruct (GPTQ 4-bit) |
| Embedding | BAAI/bge-m3 (1024-dim) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Sparse Search | BM25Okapi (rank-bm25) |
| HTML Parser | BeautifulSoup4 |
| Execution | subprocess.run() (cách ly tiến trình) |
| Data Models | Pydantic v2 |
