# Pipeline 1: OCR Text Processing, HTML Table Extraction & Hybrid Indexing

## Tổng quan

Pipeline 1 chuyển đổi **1,973 file OCR text (.txt)** chứa **146,246 bảng HTML inline** thành:
1. Kho CSV chuẩn hóa (`data/csv_warehouse/`)
2. Metadata JSON cho mỗi bảng (`data/metadata/`)
3. Chỉ mục lai Dense (BGE-M3) + Sparse (BM25) phục vụ Pipeline 2

## Sơ đồ luồng xử lý

```mermaid
flowchart TB
    subgraph INPUT ["Dữ liệu Đầu vào"]
        OCR["1,973 file .txt OCR<br/>financial_statements/TICKER/YEAR/DOC/<br/>Chứa ===== PAGE N ===== và HTML inline tables"]
        CSV_MAP["code_stock.csv<br/>Mapping Mã CK ↔ Tên công ty"]
    end

    subgraph STEP1 ["Bước 1: Page Splitter & Line Tracker"]
        S1["Đọc file .txt theo từng dòng<br/>Ghi nhận L_current (chỉ số dòng toàn cục)<br/>Phát hiện mốc ===== PAGE N =====<br/>Khi gặp thẻ &lt;table&gt; → lưu L_start"]
    end

    subgraph STEP2 ["Bước 2: HTML Table Parser"]
        S2A["Trích xuất khối &lt;table&gt;...&lt;/table&gt;<br/>bằng BeautifulSoup"]
        S2B["Thuật toán Ma trận 2D<br/>Xử lý rowspan/colspan:<br/>- 56,941 bảng có colspan (38.9%)<br/>- 45,338 bảng có rowspan (31.0%)"]
        S2C["Phẳng hóa tiêu đề MultiIndex<br/>H_level1 + '_' + H_level2 → C_unified"]
    end

    subgraph STEP3 ["Bước 3: OCR Error Correction & Normalizer"]
        S3A["Số âm ngoặc đơn: (123.456) → -123456"]
        S3B["Ký tự OCR sai: O→0, l→1, S→5, T cuối số"]
        S3C["Dấu phân cách: 1.234.567 → 1234567"]
        S3D["Detect đơn vị gốc: 'Đơn vị: VND' / 'triệu đồng'"]
    end

    subgraph STEP4 ["Bước 4: CSV Export & Metadata Builder"]
        S4A["Export DataFrame → CSV<br/>data/csv_warehouse/TICKER_YEAR_TYPE_pN_lM.csv"]
        S4B["Build JSON Metadata:<br/>doc_id, ticker, year, report_type,<br/>page_number, line_number,<br/>csv_path, columns, detected_unit,<br/>table_title"]
    end

    subgraph STEP5 ["Bước 5: Hybrid Index Builder"]
        S5A["Template-based Summary:<br/>'Công ty {ticker}, Báo cáo {type}<br/>năm {year}, Bảng {title}.<br/>Các cột: {cols}. Chỉ tiêu: {rows}'"]
        S5B["Dense Index: BAAI/bge-m3<br/>Vector embedding 1024 chiều"]
        S5C["Sparse Index: BM25Okapi<br/>Từ khóa tài chính + mã CK"]
    end

    subgraph STORAGE ["Hệ thống Lưu trữ"]
        CSV[("CSV Warehouse<br/>data/csv_warehouse/<br/>~146K file .csv")]
        META[("JSON Metadata Store<br/>data/metadata/<br/>ticker, year, type,<br/>line_number, unit")]
        DENSE[("Dense Vector Store<br/>BGE-M3 embeddings")]
        SPARSE[("Sparse BM25 Index<br/>Thuật ngữ kế toán")]
    end

    OCR --> S1
    CSV_MAP --> S4B
    S1 --> S2A --> S2B --> S2C
    S2C --> S3A --> S3B --> S3C --> S3D
    S3D --> S4A & S4B
    S4A --> CSV
    S4B --> META
    S4B --> S5A
    S5A --> S5B --> DENSE
    S5A --> S5C --> SPARSE
```

## Chi tiết thuật toán Ma trận 2D (HTML Table Parser)

```python
# Pseudocode cho thuật toán dàn phẳng ô gộp
def parse_html_table(table_html: str) -> pd.DataFrame:
    soup = BeautifulSoup(table_html, 'html.parser')
    rows = soup.find_all('tr')
    
    # 1. Tính kích thước ma trận
    max_cols = max(sum(int(td.get('colspan', 1)) for td in row.find_all(['td','th'])) for row in rows)
    matrix = [[None] * max_cols for _ in range(len(rows))]
    
    # 2. Điền giá trị vào ma trận, xử lý rowspan/colspan
    for r, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(['td', 'th']):
            while col_idx < max_cols and matrix[r][col_idx] is not None:
                col_idx += 1
            rspan = int(cell.get('rowspan', 1))
            cspan = int(cell.get('colspan', 1))
            value = cell.get_text(strip=True)
            for i in range(rspan):
                for j in range(cspan):
                    if r+i < len(matrix) and col_idx+j < max_cols:
                        matrix[r+i][col_idx+j] = value
            col_idx += cspan
    
    # 3. Tạo DataFrame từ ma trận
    return pd.DataFrame(matrix[1:], columns=matrix[0])
```

## Cấu trúc Metadata JSON cho mỗi bảng

```json
{
  "doc_id": "AAA_financial_statements_2015_consolidated",
  "ticker": "AAA",
  "year": 2015,
  "report_type": "consolidated",
  "page_number": 7,
  "line_number": 214,
  "csv_path": "data/csv_warehouse/AAA_2015_consolidated_p7_l214.csv",
  "columns": ["TÀI SẢN", "Mã số", "Thuyết minh", "31/12/2015", "01/01/2015"],
  "detected_unit": "VND",
  "table_title": "BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT",
  "num_rows": 32,
  "num_cols": 5,
  "statement_type": "CDKT"
}
```

## Thống kê Dataset thực tế

| Chỉ số | Giá trị |
|---|---|
| Tổng số file .txt | 1,973 |
| Tổng số bảng HTML | 146,246 |
| Bảng có colspan | 56,941 (38.9%) |
| Bảng có rowspan | 45,338 (31.0%) |
| Số công ty (tickers) | 100 |
| Giai đoạn | 2015–2025 |
| Dung lượng text thô | ~363 MiB |
