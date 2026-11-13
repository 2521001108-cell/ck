import json
import os
import pandas as pd
import numpy as np

BASE_DIR = r"c:\Users\Admin\Downloads\Aa"
NB_PATH = os.path.join(BASE_DIR, "Bao_Cao_Tieu_Luan_Nhom6.ipynb")
F1_PATH = r"C:/Users/Admin/Downloads/06_2521001108_PRJ04_Phan Thị Quế Trân (1).csv"
F2_PATH = r"C:/Users/Admin/Downloads/06_2521001108_PRJ04_Phan Thị Quế Trân.csv (1).csv"
OUT_HTML = os.path.join(BASE_DIR, "Dashboard_Bao_Cao_Tieu_Luan_Nhom6.html")

BAD_TOKENS = {"", "none", "nan", "n/a", "na", "not available", "?", "-"}


def extract_notebook_images(nb_path):
    """Trích xuất ảnh PNG base64 từ cell outputs của notebook."""
    images = []
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for idx, cell in enumerate(nb.get("cells", [])):
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                b64 = data["image/png"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                images.append({
                    "cell_index": idx,
                    "b64": b64.replace("\n", "").strip()
                })
    return images


def compute_dataset_stats(filepath, name):
    """Kiểm toán chất lượng dữ liệu, phân phối tài chính và thống kê khám phá."""
    df = pd.read_csv(filepath, low_memory=False, keep_default_na=False)
    n_rows = len(df)
    n_cols = len(df.columns)

    audit_rows = []
    for col in df.columns:
        s = df[col].astype(str).str.strip()
        s_low = s.str.lower()
        c_blank = int((s == "").sum())
        c_none = int((s_low == "none").sum())
        c_nan = int((s_low == "nan").sum())
        c_na = int(s_low.isin(["n/a", "na", "not available"]).sum())
        c_qm = int((s == "?").sum())
        c_dash = int((s == "-").sum())
        total_miss = c_blank + c_none + c_nan + c_na + c_qm + c_dash
        pct = round((total_miss / n_rows) * 100, 2)
        num_s = pd.to_numeric(s, errors="coerce")
        c_neg = int((num_s < 0).sum())
        audit_rows.append({
            "col": col,
            "total_miss": total_miss,
            "pct": pct,
            "blank_nan": c_blank + c_nan,
            "none": c_none,
            "na": c_na,
            "qm": c_qm,
            "dash": c_dash,
            "neg": c_neg,
        })

    def unique_clean(col_name):
        if col_name not in df.columns:
            return 0
        s = df[col_name].astype(str).str.strip()
        return int(s[~s.str.lower().isin(BAD_TOKENS)].nunique())

    # Tài chính InscClaimAmtReimbursed
    amt = pd.to_numeric(df["InscClaimAmtReimbursed"], errors="coerce")
    clean_amt = amt[amt >= 0].dropna()
    neg_amt = amt[amt < 0].dropna()
    q1 = float(clean_amt.quantile(0.25))
    med = float(clean_amt.quantile(0.50))
    q3 = float(clean_amt.quantile(0.75))
    p95 = float(clean_amt.quantile(0.95))
    p99 = float(clean_amt.quantile(0.99))
    iqr = q3 - q1
    ub = q3 + 1.5 * iqr
    outliers = clean_amt[clean_amt > ub]

    # DeductibleAmtPaid
    ded = pd.to_numeric(df["DeductibleAmtPaid"], errors="coerce")
    ded_neg = int((ded < 0).sum())

    # Phân nhóm phân phối tiền chi trả theo các khoảng giá trị
    bins = [-0.01, 50, 200, 1000, 5000, 25000, 100000, float("inf")]
    labels = ["$0–50", "$50–200", "$200–1K", "$1K–5K", "$5K–25K", "$25K–100K", "> $100K"]
    bucket_counts = pd.cut(clean_amt, bins=bins, labels=labels).value_counts().reindex(labels, fill_value=0)
    amt_buckets = [{"label": lbl, "count": int(cnt)} for lbl, cnt in bucket_counts.items()]

    # Top 10 Providers theo tổng chi trả (cho Phần 5 - Câu hỏi 1)
    prov_s = df["Provider"].astype(str).str.strip()
    valid_mask = (~prov_s.str.lower().isin(BAD_TOKENS)) & amt.notna() & (amt >= 0)
    df_valid_prov = pd.DataFrame({"Provider": prov_s[valid_mask], "Amt": amt[valid_mask]})
    top_prov = (
        df_valid_prov.groupby("Provider")["Amt"]
        .agg(claims="count", total="sum", mean="mean", max="max")
        .sort_values("total", ascending=False)
        .head(10)
        .reset_index()
    )
    top_providers = [
        {
            "provider": r["Provider"],
            "claims": int(r["claims"]),
            "total": round(float(r["total"]), 2),
            "mean": round(float(r["mean"]), 2),
            "max": round(float(r["max"]), 2),
        }
        for _, r in top_prov.iterrows()
    ]

    # Thống kê Physician overlap (Attending == Operating) cho Phần 5 - Câu hỏi 3
    att_s = df["AttendingPhysician"].astype(str).str.strip()
    op_s = df["OperatingPhysician"].astype(str).str.strip()
    both_valid = (~att_s.str.lower().isin(BAD_TOKENS)) & (~op_s.str.lower().isin(BAD_TOKENS))
    same_phys = int((att_s[both_valid] == op_s[both_valid]).sum())
    both_valid_cnt = int(both_valid.sum())

    top_op = (
        op_s[~op_s.str.lower().isin(BAD_TOKENS)]
        .value_counts()
        .head(8)
    )
    top_operating_physicians = [
        {"physician": str(k), "claims": int(v)} for k, v in top_op.items()
    ]

    # Thống kê riêng Nội trú: Length of Stay & Diagnosis_Count (Câu hỏi 2)
    los_buckets = []
    los_summary = None
    if "AdmissionDt" in df.columns and "DischargeDt" in df.columns:
        adm = pd.to_datetime(df["AdmissionDt"], errors="coerce")
        dis = pd.to_datetime(df["DischargeDt"], errors="coerce")
        los = (dis - adm).dt.days
        valid_los = los[(los >= 0) & (los <= 60)].dropna()
        diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11) if f"ClmDiagnosisCode_{i}" in df.columns]
        proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7) if f"ClmProcedureCode_{i}" in df.columns]

        def count_non_missing(cols):
            cnt = np.zeros(len(df), dtype=int)
            for c in cols:
                vals = df[c].astype(str).str.strip().str.lower()
                cnt += (~vals.isin(BAD_TOKENS)).astype(int).values
            return cnt

        diag_cnt = pd.Series(count_non_missing(diag_cols), index=df.index)
        proc_cnt = pd.Series(count_non_missing(proc_cols), index=df.index)

        los_bins = [-0.1, 1, 3, 7, 14, 30, 100]
        los_labels = ["0–1 ngày", "2–3 ngày", "4–7 ngày", "8–14 ngày", "15–30 ngày", "> 30 ngày"]
        lc = pd.cut(valid_los, bins=los_bins, labels=los_labels).value_counts().reindex(los_labels, fill_value=0)
        los_buckets = [{"label": lbl, "count": int(cnt)} for lbl, cnt in lc.items()]
        los_summary = {
            "valid_count": int(len(valid_los)),
            "mean_days": round(float(valid_los.mean()), 2),
            "median_days": round(float(valid_los.median()), 1),
            "max_days": int(valid_los.max()),
            "avg_diag_count": round(float(diag_cnt.mean()), 2),
            "avg_proc_count": round(float(proc_cnt.mean()), 2),
        }

    return {
        "name": name,
        "rows": n_rows,
        "cols": n_cols,
        "unique": {
            "bene": unique_clean("BeneID"),
            "claim": unique_clean("ClaimID"),
            "provider": unique_clean("Provider"),
            "attending": unique_clean("AttendingPhysician"),
            "operating": unique_clean("OperatingPhysician"),
        },
        "financial": {
            "valid_count": int(len(clean_amt)),
            "neg_count": int(len(neg_amt)),
            "neg_pct": round(len(neg_amt) / n_rows * 100, 2),
            "deductible_neg": ded_neg,
            "total_sum": round(float(clean_amt.sum()), 2),
            "mean": round(float(clean_amt.mean()), 2),
            "median": round(med, 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "upper_bound": round(ub, 2),
            "outlier_count": int(len(outliers)),
            "outlier_pct": round(len(outliers) / len(clean_amt) * 100, 2),
            "max": round(float(clean_amt.max()), 2),
            "buckets": amt_buckets,
        },
        "audit": audit_rows,
        "top_providers": top_providers,
        "physician_stats": {
            "both_valid_count": both_valid_cnt,
            "same_attending_operating": same_phys,
            "same_pct": round(same_phys / both_valid_cnt * 100, 2) if both_valid_cnt else 0.0,
            "top_operating": top_operating_physicians,
        },
        "los_buckets": los_buckets,
        "los_summary": los_summary,
    }


DATA_DICTIONARY = [
    (1, "BeneID", "Mã định danh người thụ hưởng bảo hiểm (Bệnh nhân)", "Chuỗi định danh", "String / Object", "Cả 2 bảng", "Khóa ngoại (FK)", "Bắt buộc có. Thiếu là lỗi dữ liệu nghiêm trọng cần loại bỏ."),
    (2, "ClaimID", "Mã hồ sơ yêu cầu chi trả bồi thường BHYT", "Chuỗi định danh", "String / Object", "Cả 2 bảng", "Khóa chính (PK)", "Định danh duy nhất cho từng giao dịch bồi thường."),
    (3, "ClaimStartDt", "Ngày bắt đầu đợt yêu cầu bồi thường BHYT", "YYYY-MM-DD", "Datetime / Object", "Cả 2 bảng", "Biến thời gian", "Bắt buộc <= ClaimEndDt."),
    (4, "ClaimEndDt", "Ngày kết thúc đợt yêu cầu bồi thường BHYT", "YYYY-MM-DD", "Datetime / Object", "Cả 2 bảng", "Biến thời gian", "Bắt buộc >= ClaimStartDt."),
    (5, "Provider", "Mã cơ sở y tế / Bệnh viện tiếp nhận điều trị", "Chuỗi định danh", "String / Object", "Cả 2 bảng", "Khóa ngoại (FK)", "Đơn vị yêu cầu bảo hiểm chi trả tiền."),
    (6, "InscClaimAmtReimbursed", "Số tiền bảo hiểm chấp thuận thanh toán bồi hoàn", "USD ($)", "Float64 / Object", "Cả 2 bảng", "Biến mục tiêu (Target)", "Tiền bảo hiểm trả. Giá trị phải >= 0 (không thể âm)."),
    (7, "DeductibleAmtPaid", "Số tiền khấu trừ người bệnh phải tự chi trả", "USD ($)", "Float64 / Object", "Cả 2 bảng", "Biến tài chính", "Tiền đồng chi trả tại viện. Giá trị phải >= 0."),
    (8, "AttendingPhysician", "Mã bác sĩ điều trị / phụ trách chuyên môn chính", "Chuỗi định danh", "String / Object", "Cả 2 bảng", "Biến nhân sự", "Bác sĩ khám lâm sàng và ký xác nhận bệnh án."),
    (9, "OperatingPhysician", "Mã bác sĩ phẫu thuật chính thực hiện can thiệp", "Chuỗi định danh", "String / Object", "Cả 2 bảng", "Biến nhân sự", "Chỉ phát sinh khi có can thiệp mổ/thủ thuật."),
    (10, "OtherPhysician", "Mã bác sĩ phụ tá / bác sĩ hội chẩn chuyên khoa", "Chuỗi định danh", "String / Object", "Cả 2 bảng", "Biến nhân sự", "Bác sĩ hỗ trợ chuyên môn cho ca bệnh phức tạp."),
    (11, "AdmissionDt", "Ngày bệnh nhân chính thức làm thủ tục nhập viện", "YYYY-MM-DD", "Datetime / Object", "Chỉ Nội trú", "Biến thời gian", "Khởi đầu đợt lưu trú giường bệnh nội trú."),
    (12, "DischargeDt", "Ngày bệnh nhân được bác sĩ ký lệnh xuất viện", "YYYY-MM-DD", "Datetime / Object", "Chỉ Nội trú", "Biến thời gian", "Dùng tính số ngày nằm viện: DischargeDt - AdmissionDt."),
    (13, "DiagnosisGroupCode", "Mã nhóm chẩn đoán liên quan đến chi phí (DRG)", "Mã nhóm DRG", "String / Object", "Chỉ Nội trú", "Biến phân loại", "Dùng để tính khung thanh toán trọn gói ca nội trú."),
    (14, "ClmAdmitDiagnosisCode", "Mã chẩn đoán bệnh sơ bộ khi vào viện tiếp nhận", "Mã ICD-9/10", "String / Object", "Cả 2 bảng", "Biến lâm sàng", "Nội trú đầy đủ 100%; Ngoại trú ít sử dụng."),
    (15, "ClmDiagnosisCode_1", "Mã chẩn đoán xác định bệnh lý chính", "Mã ICD-9/10", "String / Object", "Cả 2 bảng", "Biến lâm sàng", "Bệnh lý nguyên nhân cốt lõi dẫn đến đợt khám/điều trị."),
    (16, "ClmDiagnosisCode_2", "Mã chẩn đoán bệnh lý kèm theo thứ 2", "Mã ICD-9/10", "String / Object", "Cả 2 bảng", "Biến lâm sàng", "Bệnh nền / biến chứng đi kèm ca bệnh."),
    (17, "ClmDiagnosisCode_3", "Mã chẩn đoán bệnh lý kèm theo thứ 3", "Mã ICD-9/10", "String / Object", "Cả 2 bảng", "Biến lâm sàng", "Bệnh nền / biến chứng đi kèm ca bệnh."),
    (18, "ClmDiagnosisCode_4", "Mã chẩn đoán bệnh lý kèm theo thứ 4", "Mã ICD-9/10", "String / Object", "Cả 2 bảng", "Biến lâm sàng", "Tỷ lệ khuyết tăng dần do ít bệnh nhân đa bệnh lý."),
    (19, "ClmDiagnosisCode_5", "Mã chẩn đoán bệnh lý kèm theo thứ 5", "Mã ICD-9/10", "String / Object", "Cả 2 bảng", "Biến lâm sàng", "Tỷ lệ khuyết tăng dần do ít bệnh nhân đa bệnh lý."),
    (20, "ClmDiagnosisCode_6", "Mã chẩn đoán bệnh lý kèm theo thứ 6", "Mã ICD-9/10", "String / Object", "Cả 2 bảng", "Biến lâm sàng", "Tỷ lệ khuyết tăng dần do ít bệnh nhân đa bệnh lý."),
    (21, "ClmDiagnosisCode_7", "Mã chẩn đoán bệnh lý kèm theo thứ 7", "Mã ICD-9/10", "String / Object", "Cả 2 bảng", "Biến lâm sàng", "Tỷ lệ khuyết tăng dần do ít bệnh nhân đa bệnh lý."),
    (22, "ClmDiagnosisCode_8", "Mã chẩn đoán bệnh lý kèm theo thứ 8", "Mã ICD-9/10", "String / Object", "Cả 2 bảng", "Biến lâm sàng", "Tỷ lệ khuyết tăng dần do ít bệnh nhân đa bệnh lý."),
    (23, "ClmDiagnosisCode_9", "Mã chẩn đoán bệnh lý kèm theo thứ 9", "Mã ICD-9/10", "String / Object", "Cả 2 bảng", "Biến lâm sàng", "Tỷ lệ khuyết tăng dần do ít bệnh nhân đa bệnh lý."),
    (24, "ClmDiagnosisCode_10", "Mã chẩn đoán bệnh lý kèm theo thứ 10", "Mã ICD-9/10", "String / Object", "Cả 2 bảng", "Biến lâm sàng", "Mức độ phức tạp bệnh lý cao nhất ghi nhận được."),
    (25, "ClmProcedureCode_1", "Mã thủ thuật / can thiệp ngoại khoa thứ 1", "Mã HCPCS/ICD", "String / Object", "Cả 2 bảng", "Biến thủ thuật", "Kỹ thuật phẫu thuật chính. Ngoại trú gần như trống."),
    (26, "ClmProcedureCode_2", "Mã thủ thuật / can thiệp ngoại khoa thứ 2", "Mã HCPCS/ICD", "String / Object", "Cả 2 bảng", "Biến thủ thuật", "Kỹ thuật bổ trợ trong đợt phẫu thuật."),
    (27, "ClmProcedureCode_3", "Mã thủ thuật / can thiệp ngoại khoa thứ 3", "Mã HCPCS/ICD", "String / Object", "Cả 2 bảng", "Biến thủ thuật", "Thủ thuật thứ cấp, tỷ lệ khuyết rất cao."),
    (28, "ClmProcedureCode_4", "Mã thủ thuật / can thiệp ngoại khoa thứ 4", "Mã HCPCS/ICD", "String / Object", "Cả 2 bảng", "Biến thủ thuật", "Thủ thuật thứ cấp, tỷ lệ khuyết rất cao."),
    (29, "ClmProcedureCode_5", "Mã thủ thuật / can thiệp ngoại khoa thứ 5", "Mã HCPCS/ICD", "String / Object", "Cả 2 bảng", "Biến thủ thuật", "Thủ thuật thứ cấp, tỷ lệ khuyết rất cao."),
    (30, "ClmProcedureCode_6", "Mã thủ thuật / can thiệp ngoại khoa thứ 6", "Mã HCPCS/ICD", "String / Object", "Cả 2 bảng", "Biến thủ thuật", "Mức can thiệp ngoại khoa tối đa trong 1 hồ sơ."),
]


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Dashboard Báo Cáo Tiểu Luận Nhóm 6 — Giám Định BHYT Medicare</title>
<style>
  :root {
    --bg: #f4f6fb;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --sidebar: #0f172a;
    --sidebar-hover: #1e293b;
    --primary: #2563eb;
    --primary-soft: #eff6ff;
    --teal: #0d9488;
    --teal-soft: #f0fdfa;
    --amber: #d97706;
    --amber-soft: #fffbeb;
    --rose: #e11d48;
    --rose-soft: #fff1f2;
    --violet: #7c3aed;
    --violet-soft: #f5f3ff;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
    --radius: 14px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    min-height: 100vh;
    line-height: 1.55;
  }

  /* Sidebar */
  .sidebar {
    width: 280px;
    background: var(--sidebar);
    color: #e2e8f0;
    display: flex;
    flex-direction: column;
    position: fixed;
    top: 0; left: 0; bottom: 0;
    z-index: 40;
    border-right: 1px solid #1e293b;
  }
  .brand {
    padding: 22px 20px 18px;
    border-bottom: 1px solid #1e293b;
  }
  .brand-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: rgba(37, 99, 235, 0.22);
    color: #93c5fd;
    padding: 3px 9px;
    border-radius: 999px;
    margin-bottom: 8px;
  }
  .brand h1 {
    font-size: 16px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.35;
  }
  .brand p {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 4px;
  }
  .nav-list {
    padding: 14px 12px;
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .nav-btn {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    color: #cbd5e1;
    padding: 11px 13px;
    border-radius: 10px;
    cursor: pointer;
    display: flex;
    align-items: flex-start;
    gap: 11px;
    transition: all 0.18s ease;
    font-family: inherit;
  }
  .nav-btn:hover {
    background: var(--sidebar-hover);
    color: #ffffff;
  }
  .nav-btn.active {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #ffffff;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35);
  }
  .nav-num {
    flex-shrink: 0;
    width: 26px;
    height: 26px;
    border-radius: 8px;
    background: rgba(255,255,255,0.1);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    margin-top: 1px;
  }
  .nav-btn.active .nav-num {
    background: rgba(255,255,255,0.22);
  }
  .nav-text strong {
    display: block;
    font-size: 13.5px;
    font-weight: 600;
  }
  .nav-text span {
    display: block;
    font-size: 11.5px;
    opacity: 0.78;
    margin-top: 2px;
  }
  .sidebar-footer {
    padding: 14px 18px;
    border-top: 1px solid #1e293b;
    font-size: 12px;
    color: #94a3b8;
  }
  .sidebar-footer strong { color: #e2e8f0; }

  /* Main Area */
  .main {
    margin-left: 280px;
    flex: 1;
    padding: 26px 32px 48px;
    max-width: calc(100vw - 280px);
  }
  .topbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 24px;
    background: var(--surface);
    padding: 16px 22px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
  }
  .topbar-title h2 {
    font-size: 20px;
    font-weight: 700;
    color: var(--text);
  }
  .topbar-title p {
    font-size: 13px;
    color: var(--muted);
    margin-top: 2px;
  }
  .topbar-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .pill-btn {
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--text);
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    font-family: inherit;
  }
  .pill-btn:hover { border-color: var(--primary); color: var(--primary); }
  .pill-btn.active {
    background: var(--primary);
    color: #fff;
    border-color: var(--primary);
  }

  /* Panels */
  .section-panel { display: none; animation: fadeIn 0.22s ease; }
  .section-panel.active { display: block; }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Grid & Cards */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-bottom: 22px;
  }
  .kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
  }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: var(--primary);
  }
  .kpi-card.teal::before { background: var(--teal); }
  .kpi-card.rose::before { background: var(--rose); }
  .kpi-card.amber::before { background: var(--amber); }
  .kpi-card.violet::before { background: var(--violet); }
  .kpi-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }
  .kpi-value {
    font-size: 26px;
    font-weight: 800;
    margin: 6px 0 4px;
    color: var(--text);
  }
  .kpi-sub {
    font-size: 12.5px;
    color: var(--muted);
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
  }
  .card-header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }
  .card-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--text);
  }
  .card-subtitle {
    font-size: 13px;
    color: var(--muted);
    margin-top: 2px;
  }

  .grid-2 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
  }
  .grid-3 {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
  }

  /* Overview Part Cards */
  .part-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow);
    cursor: pointer;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .part-card:hover {
    transform: translateY(-3px);
    border-color: var(--primary);
    box-shadow: 0 10px 26px rgba(37, 99, 235, 0.12);
  }
  .part-tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--primary-soft);
    color: var(--primary);
    margin-bottom: 10px;
  }
  .part-card h3 { font-size: 16px; margin-bottom: 8px; }
  .part-card p { font-size: 13.5px; color: var(--muted); margin-bottom: 14px; }
  .part-link {
    font-size: 13px;
    font-weight: 700;
    color: var(--primary);
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  /* Badges */
  .badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 600;
    white-space: nowrap;
  }
  .badge-blue { background: var(--primary-soft); color: var(--primary); }
  .badge-teal { background: var(--teal-soft); color: var(--teal); }
  .badge-amber { background: var(--amber-soft); color: var(--amber); }
  .badge-rose { background: var(--rose-soft); color: var(--rose); }
  .badge-violet { background: var(--violet-soft); color: var(--violet); }
  .badge-slate { background: #f1f5f9; color: #334155; }

  /* Interactive Process Stepper (Part 1) */
  .stepper {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin-bottom: 18px;
  }
  .step-box {
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    cursor: pointer;
    background: var(--surface-2);
    transition: all 0.15s ease;
  }
  .step-box.active {
    border-color: var(--primary);
    background: var(--primary-soft);
  }
  .step-box h4 { font-size: 14.5px; margin-bottom: 5px; }
  .step-box p { font-size: 12.5px; color: var(--muted); }
  .step-detail-box {
    background: #f8fafc;
    border-left: 4px solid var(--primary);
    padding: 16px 20px;
    border-radius: 8px;
  }

  /* Fraud Risk Cards */
  .risk-item {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    background: var(--surface-2);
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .risk-item.active, .risk-item:hover {
    border-color: var(--rose);
    background: var(--rose-soft);
  }
  .risk-item h4 { font-size: 14.5px; color: var(--text); margin-bottom: 5px; }
  .risk-item code {
    font-size: 12px;
    background: rgba(15, 23, 42, 0.08);
    padding: 2px 6px;
    border-radius: 5px;
  }

  /* ER Schema (Part 2) */
  .er-diagram {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 18px;
    background: #f8fafc;
    border-radius: 12px;
    border: 1px dashed #cbd5e1;
  }
  .er-node {
    background: #fff;
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    min-width: 220px;
    text-align: center;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .er-node.center { border-color: var(--primary); background: var(--primary-soft); min-width: 290px; }
  .er-node:hover, .er-node.selected { border-color: var(--teal); box-shadow: 0 6px 18px rgba(13, 148, 136, 0.16); }
  .er-arrow {
    font-weight: 700;
    color: var(--muted);
    font-size: 13px;
    text-align: center;
  }

  /* Tables & Controls */
  .controls-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
  }
  .search-input {
    padding: 8px 13px;
    border: 1px solid var(--border);
    border-radius: 9px;
    font-size: 13.5px;
    min-width: 250px;
    font-family: inherit;
    outline: none;
  }
  .search-input:focus { border-color: var(--primary); }
  .filter-group {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .table-wrap {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 10px;
    max-height: 520px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  th, td {
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  th {
    background: #f1f5f9;
    font-weight: 700;
    position: sticky;
    top: 0;
    z-index: 2;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }
  th:hover { background: #e2e8f0; }
  tr:hover td { background: #f8fafc; }
  td code {
    background: #f1f5f9;
    padding: 2px 6px;
    border-radius: 5px;
    font-size: 12px;
    color: #0f172a;
    font-weight: 600;
  }
  .num-cell { text-align: right; font-variant-numeric: tabular-nums; }
  .bar-track {
    width: 90px;
    height: 8px;
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
  }
  .bar-fill { height: 100%; border-radius: 999px; }

  /* Horizontal Visual Bar Chart */
  .hbar-list { display: flex; flex-direction: column; gap: 9px; }
  .hbar-row {
    display: grid;
    grid-template-columns: 175px 1fr 72px;
    align-items: center;
    gap: 10px;
    font-size: 12.5px;
  }
  .hbar-label {
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    text-align: right;
  }
  .hbar-track {
    background: #f1f5f9;
    height: 18px;
    border-radius: 6px;
    overflow: hidden;
    position: relative;
  }
  .hbar-val {
    height: 100%;
    border-radius: 6px;
    transition: width 0.35s ease;
  }
  .hbar-pct { font-weight: 700; font-variant-numeric: tabular-nums; text-align: right; }

  /* Vertical SVG / Flex Bar Chart */
  .vchart-grid {
    display: flex;
    align-items: flex-end;
    justify-content: space-around;
    gap: 10px;
    height: 220px;
    padding: 24px 10px 8px;
    border-bottom: 2px solid var(--border);
  }
  .vbar-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    justify-content: flex-end;
  }
  .vbar-val-lbl {
    font-size: 11px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
    text-align: center;
  }
  .vbar {
    width: 72%;
    max-width: 54px;
    border-radius: 7px 7px 0 0;
    transition: height 0.35s ease;
  }
  .vbar-xlbl {
    font-size: 11.5px;
    color: var(--muted);
    margin-top: 8px;
    text-align: center;
    font-weight: 600;
  }

  /* Notebook Figure Image */
  .nb-img-wrap {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    background: #fff;
    text-align: center;
  }
  .nb-img-wrap img {
    max-width: 100%;
    height: auto;
    border-radius: 6px;
  }

  /* Sub-tabs inside sections */
  .subtabs {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
    margin-bottom: 18px;
  }
  .subtab-btn {
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--text);
    padding: 8px 15px;
    border-radius: 9px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
  }
  .subtab-btn.active {
    background: var(--text);
    color: #fff;
    border-color: var(--text);
  }

  .callout {
    padding: 14px 18px;
    border-radius: 10px;
    border-left: 4px solid var(--primary);
    background: var(--primary-soft);
    margin-bottom: 14px;
    font-size: 13.5px;
  }
  .callout.rose { border-left-color: var(--rose); background: var(--rose-soft); }
  .callout.amber { border-left-color: var(--amber); background: var(--amber-soft); }
  .callout.teal { border-left-color: var(--teal); background: var(--teal-soft); }

  @media (max-width: 900px) {
    .sidebar { position: static; width: 100%; height: auto; }
    body { flex-direction: column; }
    .main { margin-left: 0; max-width: 100%; padding: 18px; }
    .grid-2 { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<aside class="sidebar">
  <div class="brand">
    <span class="brand-badge">Nhóm 6 • Tiểu luận 2026</span>
    <h1>Giám Định BHYT Medicare</h1>
    <p>Nguồn: Bao_Cao_Tieu_Luan_Nhom6.ipynb</p>
  </div>
  <nav class="nav-list" id="navList">
    <button class="nav-btn active" data-section="overview">
      <span class="nav-num">0</span>
      <span class="nav-text">
        <strong>Tổng Quan Báo Cáo</strong>
        <span>KPIs & chọn nhanh từng phần</span>
      </span>
    </button>
    <button class="nav-btn" data-section="part1">
      <span class="nav-num">1</span>
      <span class="nav-text">
        <strong>Phần 1: Bối Cảnh & Quy Trình</strong>
        <span>Nghiệp vụ Medicare & 3 khoa/phòng</span>
      </span>
    </button>
    <button class="nav-btn" data-section="part2">
      <span class="nav-num">2</span>
      <span class="nav-text">
        <strong>Phần 2: Kiến Trúc & Từ Điển</strong>
        <span>Sơ đồ quan hệ & 30 biến chi tiết</span>
      </span>
    </button>
    <button class="nav-btn" data-section="part3">
      <span class="nav-num">3</span>
      <span class="nav-text">
        <strong>Phần 3: Đơn Vị & Cỡ Mẫu</strong>
        <span>517,737 Ngoại trú & 9,551 Nội trú</span>
      </span>
    </button>
    <button class="nav-btn" data-section="part4">
      <span class="nav-num">4</span>
      <span class="nav-text">
        <strong>Phần 4: Kiểm Toán Dữ Liệu</strong>
        <span>Bảng thiếu, số âm, Outliers & biểu đồ</span>
      </span>
    </button>
    <button class="nav-btn" data-section="part5">
      <span class="nav-num">5</span>
      <span class="nav-text">
        <strong>Phần 5: Mục Tiêu Phân Tích</strong>
        <span>3 câu hỏi kinh doanh & dữ liệu mẫu</span>
      </span>
    </button>
  </nav>
  <div class="sidebar-footer">
    <div><strong>Trưởng nhóm:</strong> Phan Thị Quế Trân</div>
    <div><strong>MSSV:</strong> 2521001108</div>
  </div>
</aside>

<main class="main">
  <header class="topbar">
    <div class="topbar-title">
      <h2 id="pageHeading">Tổng Quan Dashboard Tiểu Luận</h2>
      <p id="pageSubheading">Đề tài: Giám định và Phát hiện Bất thường trong Chi trả Bảo hiểm Y tế Medicare</p>
    </div>
    <div class="topbar-actions">
      <button class="pill-btn active" onclick="showSection('overview')">Tổng quan</button>
      <button class="pill-btn" onclick="showSection('part1')">Phần 1</button>
      <button class="pill-btn" onclick="showSection('part2')">Phần 2</button>
      <button class="pill-btn" onclick="showSection('part3')">Phần 3</button>
      <button class="pill-btn" onclick="showSection('part4')">Phần 4</button>
      <button class="pill-btn" onclick="showSection('part5')">Phần 5</button>
    </div>
  </header>

  <!-- ==================== SECTION 0: OVERVIEW ==================== -->
  <section id="sec-overview" class="section-panel active">
    <div class="kpi-grid" id="overviewKpis"></div>

    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">Bấm chọn từng phần để xem bảng thông tin trực quan chi tiết</div>
          <div class="card-subtitle">Mỗi thẻ dưới đây tương ứng với 1 chương/phần trong file Bao_Cao_Tieu_Luan_Nhom6.ipynb</div>
        </div>
      </div>
      <div class="grid-3">
        <div class="part-card" onclick="showSection('part1')">
          <div>
            <span class="part-tag">PHẦN 1</span>
            <h3>Bối cảnh kinh doanh &amp; Quy trình phát sinh dữ liệu</h3>
            <p>4 hành vi trục lợi BHYT trọng tâm, phân biệt Ngoại trú vs Nội trú và chuỗi quy trình 3 phòng ban.</p>
          </div>
          <span class="part-link">Xem chi tiết Phần 1 &rarr;</span>
        </div>
        <div class="part-card" onclick="showSection('part2')">
          <div>
            <span class="part-tag">PHẦN 2</span>
            <h3>Kiến trúc &amp; Từ điển dữ liệu (30 biến)</h3>
            <p>Mô hình quan hệ PK/FK (ClaimID, BeneID, Provider) và bảng tra cứu tương tác 30 thuộc tính.</p>
          </div>
          <span class="part-link">Xem chi tiết Phần 2 &rarr;</span>
        </div>
        <div class="part-card" onclick="showSection('part3')">
          <div>
            <span class="part-tag">PHẦN 3</span>
            <h3>Đơn vị quan sát &amp; Cỡ mẫu dữ liệu</h3>
            <p>So sánh quy mô Ngoại trú (517,737 dòng × 27 cột) và Nội trú (9,551 dòng × 30 cột) cùng số lượng thực thể duy nhất.</p>
          </div>
          <span class="part-link">Xem chi tiết Phần 3 &rarr;</span>
        </div>
        <div class="part-card" onclick="showSection('part4')">
          <div>
            <span class="part-tag" style="background:var(--rose-soft);color:var(--rose);">PHẦN 4</span>
            <h3>Kiểm toán Độ sạch Dữ liệu (Data Quality Audit)</h3>
            <p>Bảng thống kê thiếu dữ liệu từng cột, ký tự nhiễu (None, N/A, ?, -), giá trị âm, phân tích Outliers và biểu đồ gốc.</p>
          </div>
          <span class="part-link" style="color:var(--rose);">Xem chi tiết Phần 4 &rarr;</span>
        </div>
        <div class="part-card" onclick="showSection('part5')">
          <div>
            <span class="part-tag" style="background:var(--teal-soft);color:var(--teal);">PHẦN 5</span>
            <h3>Mục tiêu Phân tích &amp; 3 Câu hỏi Kinh doanh</h3>
            <p>Phát hiện Provider kê khống, phân tích ngày nằm viện (Length of Stay) và hồ sơ y bác sĩ, kèm số liệu thực tế từ CSV.</p>
          </div>
          <span class="part-link" style="color:var(--teal);">Xem chi tiết Phần 5 &rarr;</span>
        </div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">So sánh nhanh 3 nhóm rủi ro dữ liệu (Ngoại trú vs Nội trú)</div>
            <div class="card-subtitle">Tổng hợp trực tiếp từ kết quả kiểm toán Python</div>
          </div>
        </div>
        <div id="overviewRiskBars"></div>
      </div>
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Tóm tắt quy mô tài chính hợp lệ (InscClaimAmtReimbursed &ge; 0)</div>
            <div class="card-subtitle">So sánh tổng tiền bồi thường và ngưỡng ngoại lai (IQR)</div>
          </div>
        </div>
        <div id="overviewFinanceTable"></div>
      </div>
    </div>
  </section>

  <!-- ==================== SECTION 1: PART 1 ==================== -->
  <section id="sec-part1" class="section-panel">
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">1.1. Bốn hành vi gian lận &amp; trục lợi BHYT Medicare trọng tâm</div>
          <div class="card-subtitle">Bấm vào từng thẻ hành vi để xem dấu hiệu nhận biết và biến dữ liệu liên quan</div>
        </div>
      </div>
      <div class="grid-2" id="fraudCards"></div>
      <div class="callout rose" id="fraudDetail"></div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div class="card-title">So sánh hai nghiệp vụ khám chữa bệnh</div>
        </div>
        <table>
          <thead>
            <tr><th>Tiêu chí</th><th>Ngoại trú (Outpatient)</th><th>Nội trú (Inpatient)</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>Hình thức điều trị</strong></td><td>Khám lâm sàng, xét nghiệm, chụp chiếu, nhận thuốc và về trong ngày</td><td>Nhập viện lưu trú qua đêm để điều trị nội khoa hoặc phẫu thuật</td></tr>
            <tr><td><strong>Lưu trú giường bệnh</strong></td><td><span class="badge badge-slate">Không phát sinh</span></td><td><span class="badge badge-blue">Từ 1 ngày đến nhiều tuần</span></td></tr>
            <tr><td><strong>Biến đặc thù</strong></td><td>Không có ngày nhập/xuất viện &amp; DRG</td><td>Có <code>AdmissionDt</code>, <code>DischargeDt</code>, <code>DiagnosisGroupCode</code></td></tr>
            <tr><td><strong>Quy mô dữ liệu</strong></td><td>517,737 hồ sơ (27 cột)</td><td>9,551 hồ sơ (30 cột)</td></tr>
          </tbody>
        </table>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">1.2. Chuỗi quy trình phát sinh dữ liệu (3 Phòng ban)</div>
            <div class="card-subtitle">Bấm chọn từng bước để xem chi tiết nghiệp vụ và các cột được tạo ra</div>
          </div>
        </div>
        <div class="stepper" id="processStepper"></div>
        <div class="step-detail-box" id="processDetail"></div>
      </div>
    </div>
  </section>

  <!-- ==================== SECTION 2: PART 2 ==================== -->
  <section id="sec-part2" class="section-panel">
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Tổng số biến chuẩn hóa</div>
        <div class="kpi-value">30 biến</div>
        <div class="kpi-sub">Đầy đủ 27 biến chung + 3 biến Nội trú</div>
      </div>
      <div class="kpi-card teal">
        <div class="kpi-label">Khóa chính &amp; Khóa ngoại</div>
        <div class="kpi-value">3 khóa</div>
        <div class="kpi-sub">PK: ClaimID | FK: BeneID, Provider</div>
      </div>
      <div class="kpi-card rose">
        <div class="kpi-label">Biến mục tiêu (Target)</div>
        <div class="kpi-value">1 biến</div>
        <div class="kpi-sub">InscClaimAmtReimbursed (USD)</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-label">Biến chỉ có ở Nội trú</div>
        <div class="kpi-value">3 biến</div>
        <div class="kpi-sub">AdmissionDt, DischargeDt, DiagnosisGroupCode</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">2.1. Kiến trúc quan hệ thực thể (Relational Keys)</div>
          <div class="card-subtitle">Bấm vào từng khối bảng để xem vai trò liên kết</div>
        </div>
      </div>
      <div class="er-diagram">
        <div class="er-node" onclick="selectErNode('bene', this)">
          <span class="badge badge-teal">Thực thể Bệnh nhân (1)</span>
          <h4 style="margin:6px 0;">Bảng Nhân khẩu học</h4>
          <p style="font-size:12.5px;color:var(--muted);">Khóa liên kết: <code>BeneID</code></p>
        </div>
        <div class="er-arrow">1 &mdash; N<br/>(BeneID)</div>
        <div class="er-node center selected" onclick="selectErNode('claims', this)">
          <span class="badge badge-blue">Bảng Giao dịch Trung tâm (N)</span>
          <h4 style="margin:6px 0;">HỒ SƠ BỒI THƯỜNG BHYT</h4>
          <p style="font-size:12.5px;color:var(--muted);">Khóa chính (PK): <code>ClaimID</code><br/>Ngoại trú: 517,737 dòng | Nội trú: 9,551 dòng</p>
        </div>
        <div class="er-arrow">N &mdash; 1<br/>(Provider)</div>
        <div class="er-node" onclick="selectErNode('provider', this)">
          <span class="badge badge-violet">Thực thể Cơ sở Y tế (1)</span>
          <h4 style="margin:6px 0;">Danh mục Bệnh viện / Phòng khám</h4>
          <p style="font-size:12.5px;color:var(--muted);">Khóa liên kết: <code>Provider</code></p>
        </div>
      </div>
      <div class="callout" id="erDetail" style="margin-top:14px;margin-bottom:0;"></div>
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">2.2. Bảng Từ điển Dữ liệu Chi tiết (30 Biến)</div>
          <div class="card-subtitle">Tìm kiếm theo tên biến hoặc lọc theo vai trò nghiệp vụ</div>
        </div>
      </div>
      <div class="controls-bar">
        <input type="text" class="search-input" id="dictSearch" placeholder="Tìm tên biến hoặc ý nghĩa (vd: ClaimID, ICD, USD)..." oninput="renderDictionary()" />
        <div class="filter-group" id="dictFilters"></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>STT</th>
              <th>Tên Biến</th>
              <th>Ý Nghĩa Thực Tế Nghiệp Vụ</th>
              <th>Đơn Vị / Định Dạng</th>
              <th>Kiểu Dữ Liệu</th>
              <th>Bảng</th>
              <th>Vai Trò</th>
              <th>Tính Chất &amp; Ràng Buộc</th>
            </tr>
          </thead>
          <tbody id="dictBody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- ==================== SECTION 3: PART 3 ==================== -->
  <section id="sec-part3" class="section-panel">
    <div class="callout">
      <strong>3.1. Đơn vị quan sát (Unit of Observation):</strong> Mỗi dòng (record) trong cả hai bộ dữ liệu đại diện cho <strong>Một giao dịch yêu cầu bồi thường bảo hiểm y tế duy nhất (One Healthcare Insurance Claim Event)</strong> gắn liền với một đợt khám/chữa bệnh của một bệnh nhân tại một cơ sở y tế xác định.
    </div>

    <div class="kpi-grid" id="part3Kpis"></div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">So sánh quy mô thực thể duy nhất (Unique Entities)</div>
            <div class="card-subtitle">Tính trực tiếp trên các giá trị hợp lệ của 2 file CSV</div>
          </div>
        </div>
        <div id="part3EntityChart"></div>
      </div>
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Bảng tổng hợp Cỡ mẫu &amp; Độ phủ định danh</div>
            <div class="card-subtitle">Số dòng, số cột và số thực thể duy nhất theo từng bảng</div>
          </div>
        </div>
        <div id="part3SummaryTable"></div>
      </div>
    </div>
  </section>

  <!-- ==================== SECTION 4: PART 4 ==================== -->
  <section id="sec-part4" class="section-panel">
    <div class="card" style="padding:16px 22px;">
      <div class="controls-bar" style="margin-bottom:0;">
        <div>
          <strong style="font-size:15px;">Chọn chế độ kiểm toán &amp; trực quan hóa:</strong>
          <span style="font-size:13px;color:var(--muted);margin-left:8px;">Bấm chọn để xem bảng số liệu chi tiết hoặc hình ảnh biểu đồ gốc</span>
        </div>
        <div class="filter-group">
          <button class="pill-btn active" id="btnAuditOut" onclick="switchAuditDataset('outpatient')">Ngoại Trú (517,737 dòng)</button>
          <button class="pill-btn" id="btnAuditIn" onclick="switchAuditDataset('inpatient')">Nội Trú (9,551 dòng)</button>
          <button class="pill-btn" id="btnAuditCompare" onclick="switchAuditDataset('compare')">So sánh Cả Hai</button>
          <button class="pill-btn" id="btnAuditImages" style="background:#4f46e5;color:#fff;border-color:#4f46e5;" onclick="switchAuditDataset('images')">📷 Xem 2 Biểu Đồ Hình Ảnh Gốc</button>
        </div>
      </div>
    </div>

    <div id="auditDynamicContent"></div>

    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">4.1. Kết luận Kiểm toán Độ sạch Ban đầu (3 Nhóm rủi ro lớn)</div>
          <div class="card-subtitle">Tổng hợp phát hiện từ mã kiểm toán Python trên cả hai bộ dữ liệu</div>
        </div>
      </div>
      <div class="grid-3">
        <div class="callout amber" style="margin-bottom:0;">
          <strong>1. Nhiễm bẩn dữ liệu thiếu (Dirty Missing)</strong>
          <p style="margin-top:6px;">Dữ liệu thiếu xuất hiện dưới dạng chuỗi nhân tạo: <code>'None'</code>, <code>'N/A'</code>, <code>'?'</code>, <code>'-'</code>, <code>''</code>. Cần chuẩn hóa toàn bộ về <code>np.nan</code> trước khi ép kiểu.</p>
        </div>
        <div class="callout" style="margin-bottom:0;">
          <strong>2. Khuyết thiếu Khóa định danh (Identity Leakage)</strong>
          <p style="margin-top:6px;">Các cột định danh <code>BeneID</code>, <code>ClaimID</code>, <code>Provider</code> bị thiếu từ ~5.5% đến ~15.0%. Không được phép điền khuyết mà phải loại bỏ để bảo toàn tính toàn vẹn giao dịch.</p>
        </div>
        <div class="callout rose" style="margin-bottom:0;">
          <strong>3. Dị thường số học &amp; Ngoại lai cực đoan</strong>
          <p style="margin-top:6px;">Phát hiện <strong>7,344 giá trị âm</strong> ở Ngoại trú và <strong>346 giá trị âm</strong> ở Nội trú tại <code>InscClaimAmtReimbursed</code>. Phân phối lệch phải mạnh với nhiều Outliers vượt ngưỡng IQR.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- ==================== SECTION 5: PART 5 ==================== -->
  <section id="sec-part5" class="section-panel">
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">Mục tiêu Phân tích &amp; 3 Câu hỏi Kinh doanh Thực tế</div>
          <div class="card-subtitle">Bấm chọn từng câu hỏi để xem phương pháp Python, giá trị kinh doanh và số liệu khám phá sơ bộ từ dữ liệu</div>
        </div>
      </div>
      <div class="subtabs">
        <button class="subtab-btn active" onclick="switchQTab('q1', this)">Câu hỏi 1: Nhận diện Provider trục lợi</button>
        <button class="subtab-btn" onclick="switchQTab('q2', this)">Câu hỏi 2: Thời gian nằm viện &amp; Độ phức tạp</button>
        <button class="subtab-btn" onclick="switchQTab('q3', this)">Câu hỏi 3: Hồ sơ &amp; Mạng lưới Y bác sĩ</button>
      </div>
      <div id="qContent"></div>
    </div>
  </section>
</main>

<script>
const REPORT_DATA = __REPORT_DATA_JSON__;

const SECTION_META = {
  overview: {
    title: "Tổng Quan Dashboard Tiểu Luận",
    sub: "Đề tài: Giám định và Phát hiện Bất thường trong Chi trả Bảo hiểm Y tế Medicare"
  },
  part1: {
    title: "Phần 1: Bối Cảnh Kinh Doanh & Quy Trình Phát Sinh Dữ Liệu",
    sub: "Nghiệp vụ bảo hiểm Medicare, 4 hành vi gian lận phổ biến và quy trình 3 phòng ban"
  },
  part2: {
    title: "Phần 2: Kiến Trúc & Từ Điển Dữ Liệu (30 Biến)",
    sub: "Sơ đồ khóa liên kết (ClaimID, BeneID, Provider) và bảng tra cứu 30 thuộc tính"
  },
  part3: {
    title: "Phần 3: Đơn Vị Quan Sát & Cỡ Mẫu Dữ Liệu",
    sub: "Quy mô 517,737 hồ sơ Ngoại trú (27 cột) & 9,551 hồ sơ Nội trú (30 cột)"
  },
  part4: {
    title: "Phần 4: Thực Thi Kiểm Toán Độ Sạch Ban Đầu (Data Quality Audit)",
    sub: "Bảng kiểm toán chi tiết từng cột, phân tích ký tự thiếu, giá trị âm & Outliers"
  },
  part5: {
    title: "Phần 5: Mục Tiêu Phân Tích (3 Câu Hỏi Kinh Doanh)",
    sub: "Phương pháp Python đề xuất & thống kê khám phá sơ bộ từ 2 tập dữ liệu CSV"
  }
};

function fmtNum(n) {
  return Number(n).toLocaleString("en-US");
}
function fmtUSD(n) {
  return "$" + Number(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function fmtCompactUSD(n) {
  if (n >= 1e9) return "$" + (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return "$" + (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return "$" + (n / 1e3).toFixed(1) + "K";
  return fmtUSD(n);
}

function showSection(secId) {
  document.querySelectorAll(".section-panel").forEach(p => p.classList.remove("active"));
  document.getElementById("sec-" + secId).classList.add("active");

  document.querySelectorAll(".nav-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.section === secId);
  });
  const topOrder = ["overview", "part1", "part2", "part3", "part4", "part5"];
  document.querySelectorAll(".topbar-actions .pill-btn").forEach((b, i) => {
    b.classList.toggle("active", topOrder[i] === secId);
  });

  document.getElementById("pageHeading").textContent = SECTION_META[secId].title;
  document.getElementById("pageSubheading").textContent = SECTION_META[secId].sub;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => showSection(btn.dataset.section));
});

/* ==================== OVERVIEW ==================== */
function renderOverview() {
  const out = REPORT_DATA.outpatient;
  const inp = REPORT_DATA.inpatient;
  const totRows = out.rows + inp.rows;
  const totSum = out.financial.total_sum + inp.financial.total_sum;
  const totNeg = out.financial.neg_count + inp.financial.neg_count;
  const totOutliers = out.financial.outlier_count + inp.financial.outlier_count;

  document.getElementById("overviewKpis").innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">Tổng hồ sơ bồi thường</div>
      <div class="kpi-value">${fmtNum(totRows)}</div>
      <div class="kpi-sub">Ngoại trú: ${fmtNum(out.rows)} | Nội trú: ${fmtNum(inp.rows)}</div>
    </div>
    <div class="kpi-card teal">
      <div class="kpi-label">Tổng chi trả hợp lệ</div>
      <div class="kpi-value">${fmtCompactUSD(totSum)}</div>
      <div class="kpi-sub">Ngoại trú ${fmtCompactUSD(out.financial.total_sum)} | Nội trú ${fmtCompactUSD(inp.financial.total_sum)}</div>
    </div>
    <div class="kpi-card rose">
      <div class="kpi-label">Dị thường số âm (&lt; 0)</div>
      <div class="kpi-value">${fmtNum(totNeg)}</div>
      <div class="kpi-sub">Cột InscClaimAmtReimbursed (Ngoại: ${fmtNum(out.financial.neg_count)}, Nội: ${fmtNum(inp.financial.neg_count)})</div>
    </div>
    <div class="kpi-card amber">
      <div class="kpi-label">Hồ sơ Ngoại lai (IQR)</div>
      <div class="kpi-value">${fmtNum(totOutliers)}</div>
      <div class="kpi-sub">Ngoại trú: ${out.financial.outlier_pct}% | Nội trú: ${inp.financial.outlier_pct}%</div>
    </div>
  `;

  // Compare key missing rates
  const compareCols = ["BeneID", "ClaimID", "Provider", "InscClaimAmtReimbursed", "AttendingPhysician", "OperatingPhysician"];
  const outMap = Object.fromEntries(out.audit.map(r => [r.col, r.pct]));
  const inMap = Object.fromEntries(inp.audit.map(r => [r.col, r.pct]));

  let riskHtml = `<div class="hbar-list">`;
  compareCols.forEach(c => {
    const op = outMap[c] || 0;
    const ip = inMap[c] || 0;
    riskHtml += `
      <div style="margin-bottom:6px;">
        <div style="display:flex;justify-content:space-between;font-size:12.5px;font-weight:600;margin-bottom:3px;">
          <span><code>${c}</code></span>
          <span>Ngoại trú: <strong style="color:#e11d48">${op}%</strong> &nbsp;|&nbsp; Nội trú: <strong style="color:#2563eb">${ip}%</strong></span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
          <div class="hbar-track"><div class="hbar-val" style="width:${op}%;background:#f43f5e;"></div></div>
          <div class="hbar-track"><div class="hbar-val" style="width:${ip}%;background:#3b82f6;"></div></div>
        </div>
      </div>`;
  });
  riskHtml += `</div>`;
  document.getElementById("overviewRiskBars").innerHTML = riskHtml;

  document.getElementById("overviewFinanceTable").innerHTML = `
    <table>
      <thead>
        <tr><th>Chỉ số (InscClaimAmtReimbursed)</th><th class="num-cell">Ngoại trú</th><th class="num-cell">Nội trú</th></tr>
      </thead>
      <tbody>
        <tr><td>Bản ghi hợp lệ (&ge; 0)</td><td class="num-cell">${fmtNum(out.financial.valid_count)}</td><td class="num-cell">${fmtNum(inp.financial.valid_count)}</td></tr>
        <tr><td>Giá trị âm (&lt; 0)</td><td class="num-cell" style="color:var(--rose);font-weight:700;">${fmtNum(out.financial.neg_count)} (${out.financial.neg_pct}%)</td><td class="num-cell" style="color:var(--rose);font-weight:700;">${fmtNum(inp.financial.neg_count)} (${inp.financial.neg_pct}%)</td></tr>
        <tr><td>Trung bình (Mean)</td><td class="num-cell">${fmtUSD(out.financial.mean)}</td><td class="num-cell">${fmtUSD(inp.financial.mean)}</td></tr>
        <tr><td>Trung vị (Median)</td><td class="num-cell">${fmtUSD(out.financial.median)}</td><td class="num-cell">${fmtUSD(inp.financial.median)}</td></tr>
        <tr><td>Ngưỡng Outlier trên (Q3 + 1.5*IQR)</td><td class="num-cell">${fmtUSD(out.financial.upper_bound)}</td><td class="num-cell">${fmtUSD(inp.financial.upper_bound)}</td></tr>
        <tr><td>Số lượng Outliers</td><td class="num-cell">${fmtNum(out.financial.outlier_count)} (${out.financial.outlier_pct}%)</td><td class="num-cell">${fmtNum(inp.financial.outlier_count)} (${inp.financial.outlier_pct}%)</td></tr>
        <tr><td>Chi trả cao nhất (Max)</td><td class="num-cell" style="font-weight:700;">${fmtUSD(out.financial.max)}</td><td class="num-cell" style="font-weight:700;">${fmtUSD(inp.financial.max)}</td></tr>
      </tbody>
    </table>
  `;
}

/* ==================== PART 1 ==================== */
const FRAUD_ITEMS = [
  {
    title: "1. Kê khống viện phí",
    col: "InscClaimAmtReimbursed",
    desc: "Yêu cầu bồi hoàn số tiền lớn hơn thực tế chi phí điều trị.",
    detail: "Cơ sở y tế khai tăng chi phí trong cột <code>InscClaimAmtReimbursed</code>. Đây là biến mục tiêu trọng tâm (Target Variable), nơi xuất hiện các giá trị Outliers lên đến hàng triệu USD trong tập dữ liệu."
  },
  {
    title: "2. Gian lận dịch vụ y tế",
    col: "ClmProcedureCode_1..6",
    desc: "Chỉ định các thủ thuật phẫu thuật phức tạp không cần thiết.",
    detail: "Kiểm tra các mã thủ thuật <code>ClmProcedureCode_1</code> đến <code>_6</code> và đối chiếu với bác sĩ phẫu thuật <code>OperatingPhysician</code> để phát hiện chỉ định thừa hoặc bất hợp lý."
  },
  {
    title: "3. Lạm dụng mã chẩn đoán (Upcoding)",
    col: "ClmDiagnosisCode_1..10",
    desc: "Gán thêm mã chẩn đoán bệnh lý nặng để nâng khung thanh toán.",
    detail: "Gán nhiều mã bệnh ICD-9/10 từ <code>ClmDiagnosisCode_1</code> đến <code>_10</code> hoặc nâng mã <code>DiagnosisGroupCode</code> (DRG) ở Nội trú để hưởng mức chi trả trọn gói cao hơn."
  },
  {
    title: "4. Kéo dài ngày nằm viện",
    col: "Length of Stay (AdmissionDt → DischargeDt)",
    desc: "Giữ bệnh nhân nội trú lâu hơn mức cần thiết để thu tiền giường.",
    detail: "Tính toán <code>LengthOfStay = (DischargeDt - AdmissionDt).dt.days</code> trên bảng Nội trú và đối chiếu với số lượng mã bệnh/thủ thuật thực tế."
  }
];

const PROCESS_STEPS = [
  {
    step: "Bước 1: Phòng Tiếp đón & Đăng ký Khám",
    short: "Admissions / Front Desk",
    detail: "Tiếp nhận thẻ BHYT, định danh bệnh nhân bằng <code>BeneID</code> và ghi nhận ngày bắt đầu khám (<code>ClaimStartDt</code>) hoặc ngày chính thức nhập viện nội trú (<code>AdmissionDt</code>)."
  },
  {
    step: "Bước 2: Khoa Khám bệnh & Lâm sàng / Phẫu thuật",
    short: "Clinical & Surgical Departments",
    detail: "Bác sĩ điều trị chính (<code>AttendingPhysician</code>) ghi mã chẩn đoán vào viện (<code>ClmAdmitDiagnosisCode</code>) &amp; chẩn đoán xác định (<code>ClmDiagnosisCode_1..10</code>). Nếu phẫu thuật, <code>OperatingPhysician</code> &amp; <code>OtherPhysician</code> ghi mã <code>ClmProcedureCode_1..6</code> và ký xuất viện (<code>DischargeDt</code>)."
  },
  {
    step: "Bước 3: Phòng Kế toán Viện phí & Giám định BHYT",
    short: "Billing & Medical Coding",
    detail: "Tổng hợp bảng kê, tính tiền khấu trừ bệnh nhân tự trả (<code>DeductibleAmtPaid</code>), tạo khóa chính <code>ClaimID</code>, gắn mã cơ sở <code>Provider</code> và gửi hồ sơ yêu cầu chi trả <code>InscClaimAmtReimbursed</code> lên cổng CMS."
  }
];

function renderPart1() {
  const fc = document.getElementById("fraudCards");
  fc.innerHTML = FRAUD_ITEMS.map((item, i) => `
    <div class="risk-item ${i === 0 ? 'active' : ''}" onclick="selectFraud(${i}, this)">
      <h4>${item.title}</h4>
      <p style="font-size:13px;color:var(--muted);margin-bottom:6px;">${item.desc}</p>
      <code>${item.col}</code>
    </div>
  `).join("");
  document.getElementById("fraudDetail").innerHTML = `<strong>Chi tiết giám định:</strong> ${FRAUD_ITEMS[0].detail}`;

  const ps = document.getElementById("processStepper");
  ps.innerHTML = PROCESS_STEPS.map((s, i) => `
    <div class="step-box ${i === 0 ? 'active' : ''}" onclick="selectProcess(${i}, this)">
      <h4>${s.step}</h4>
      <p>${s.short}</p>
    </div>
  `).join("");
  document.getElementById("processDetail").innerHTML = `<strong>Nghiệp vụ &amp; Dữ liệu phát sinh:</strong> ${PROCESS_STEPS[0].detail}`;
}

function selectFraud(idx, el) {
  document.querySelectorAll("#fraudCards .risk-item").forEach(x => x.classList.remove("active"));
  el.classList.add("active");
  document.getElementById("fraudDetail").innerHTML = `<strong>Chi tiết giám định:</strong> ${FRAUD_ITEMS[idx].detail}`;
}

function selectProcess(idx, el) {
  document.querySelectorAll("#processStepper .step-box").forEach(x => x.classList.remove("active"));
  el.classList.add("active");
  document.getElementById("processDetail").innerHTML = `<strong>Nghiệp vụ &amp; Dữ liệu phát sinh:</strong> ${PROCESS_STEPS[idx].detail}`;
}

/* ==================== PART 2 ==================== */
const ER_DETAILS = {
  bene: "<strong>BeneID (Foreign Key):</strong> Liên kết 1–N từ Bảng Nhân khẩu học Bệnh nhân tới Hồ sơ Bồi thường. Một bệnh nhân có thể phát sinh nhiều đợt khám ngoại trú hoặc nhập viện nội trú khác nhau.",
  claims: "<strong>ClaimID (Primary Key):</strong> Định danh duy nhất cho mỗi giao dịch yêu cầu bồi thường. Bảng Ngoại trú có 27 cột (517,737 dòng), bảng Nội trú có 30 cột (9,551 dòng — bổ sung <code>AdmissionDt</code>, <code>DischargeDt</code>, <code>DiagnosisGroupCode</code>).",
  provider: "<strong>Provider (Foreign Key):</strong> Liên kết 1–N từ Danh mục Cơ sở Y tế tới Hồ sơ Bồi thường. Đây là đối tượng trọng tâm để tổng hợp dòng tiền và phát hiện dấu hiệu trục lợi BHYT."
};

let dictFilterRole = "Tất cả";

function selectErNode(key, el) {
  document.querySelectorAll(".er-node").forEach(n => n.classList.remove("selected"));
  el.classList.add("selected");
  document.getElementById("erDetail").innerHTML = ER_DETAILS[key];
}

function initPart2() {
  document.getElementById("erDetail").innerHTML = ER_DETAILS.claims;
  const roles = ["Tất cả", "Khóa", "Target & Tài chính", "Thời gian", "Lâm sàng", "Thủ thuật", "Nhân sự", "Chỉ Nội trú"];
  document.getElementById("dictFilters").innerHTML = roles.map((r, i) => `
    <button class="pill-btn ${i === 0 ? 'active' : ''}" onclick="setDictRole('${r}', this)">${r}</button>
  `).join("");
  renderDictionary();
}

function setDictRole(role, el) {
  dictFilterRole = role;
  document.querySelectorAll("#dictFilters .pill-btn").forEach(b => b.classList.remove("active"));
  el.classList.add("active");
  renderDictionary();
}

function renderDictionary() {
  const q = (document.getElementById("dictSearch").value || "").trim().toLowerCase();
  const rows = REPORT_DATA.dictionary.filter(item => {
    const matchText = !q || item.name.toLowerCase().includes(q) || item.meaning.toLowerCase().includes(q) || item.note.toLowerCase().includes(q);
    if (!matchText) return false;
    if (dictFilterRole === "Tất cả") return true;
    if (dictFilterRole === "Khóa") return item.role.includes("Khóa");
    if (dictFilterRole === "Target & Tài chính") return item.role.includes("mục tiêu") || item.role.includes("tài chính");
    if (dictFilterRole === "Thời gian") return item.role.includes("thời gian");
    if (dictFilterRole === "Lâm sàng") return item.role.includes("lâm sàng") || item.role.includes("phân loại");
    if (dictFilterRole === "Thủ thuật") return item.role.includes("thủ thuật");
    if (dictFilterRole === "Nhân sự") return item.role.includes("nhân sự");
    if (dictFilterRole === "Chỉ Nội trú") return item.table === "Chỉ Nội trú";
    return true;
  });

  document.getElementById("dictBody").innerHTML = rows.map(r => {
    const tblBadge = r.table === "Chỉ Nội trú"
      ? `<span class="badge badge-amber">Chỉ Nội trú</span>`
      : `<span class="badge badge-slate">Cả 2 bảng</span>`;
    let roleClass = "badge-slate";
    if (r.role.includes("Khóa")) roleClass = "badge-blue";
    else if (r.role.includes("mục tiêu")) roleClass = "badge-rose";
    else if (r.role.includes("tài chính")) roleClass = "badge-teal";
    else if (r.role.includes("nhân sự")) roleClass = "badge-violet";
    return `
      <tr>
        <td><strong>${r.stt}</strong></td>
        <td><code>${r.name}</code></td>
        <td>${r.meaning}</td>
        <td>${r.unit}</td>
        <td>${r.dtype}</td>
        <td>${tblBadge}</td>
        <td><span class="badge ${roleClass}">${r.role}</span></td>
        <td style="font-size:12.5px;color:var(--muted);">${r.note}</td>
      </tr>`;
  }).join("");
}

/* ==================== PART 3 ==================== */
function renderPart3() {
  const out = REPORT_DATA.outpatient;
  const inp = REPORT_DATA.inpatient;

  document.getElementById("part3Kpis").innerHTML = `
    <div class="kpi-card">
      <div class="kpi-label">Bảng 1: Ngoại trú (Outpatient)</div>
      <div class="kpi-value">${fmtNum(out.rows)} dòng</div>
      <div class="kpi-sub">${out.cols} cột thuộc tính • Tỷ lệ: ${(out.rows / (out.rows + inp.rows) * 100).toFixed(2)}% tổng số dòng</div>
    </div>
    <div class="kpi-card teal">
      <div class="kpi-label">Bảng 2: Nội trú (Inpatient)</div>
      <div class="kpi-value">${fmtNum(inp.rows)} dòng</div>
      <div class="kpi-sub">${inp.cols} cột thuộc tính • Tỷ lệ: ${(inp.rows / (out.rows + inp.rows) * 100).toFixed(2)}% tổng số dòng</div>
    </div>
    <div class="kpi-card violet">
      <div class="kpi-label">Bệnh nhân duy nhất (BeneID)</div>
      <div class="kpi-value">${fmtNum(out.unique.bene)} / ${fmtNum(inp.unique.bene)}</div>
      <div class="kpi-sub">Ngoại trú: ${fmtNum(out.unique.bene)} | Nội trú: ${fmtNum(inp.unique.bene)}</div>
    </div>
    <div class="kpi-card amber">
      <div class="kpi-label">Cơ sở Y tế (Provider)</div>
      <div class="kpi-value">${fmtNum(out.unique.provider)} / ${fmtNum(inp.unique.provider)}</div>
      <div class="kpi-sub">Ngoại trú: ${fmtNum(out.unique.provider)} | Nội trú: ${fmtNum(inp.unique.provider)}</div>
    </div>
  `;

  const metrics = [
    { label: "Tổng dòng dữ liệu", out: out.rows, inp: inp.rows },
    { label: "ClaimID duy nhất (hợp lệ)", out: out.unique.claim, inp: inp.unique.claim },
    { label: "BeneID duy nhất", out: out.unique.bene, inp: inp.unique.bene },
    { label: "AttendingPhysician duy nhất", out: out.unique.attending, inp: inp.unique.attending },
    { label: "OperatingPhysician duy nhất", out: out.unique.operating, inp: inp.unique.operating },
    { label: "Provider duy nhất", out: out.unique.provider, inp: inp.unique.provider }
  ];

  let chartHtml = `<div class="hbar-list">`;
  metrics.forEach(m => {
    const outW = Math.max(4, Math.round((m.out / out.rows) * 100));
    const inW = Math.max(4, Math.round((m.inp / inp.rows) * 100));
    chartHtml += `
      <div style="padding-bottom:6px;border-bottom:1px solid #f1f5f9;">
        <div style="font-size:12.5px;font-weight:700;margin-bottom:4px;">${m.label}</div>
        <div class="hbar-row" style="grid-template-columns:90px 1fr 85px;">
          <span class="hbar-label" style="color:var(--primary);">Ngoại trú</span>
          <div class="hbar-track"><div class="hbar-val" style="width:${outW}%;background:#3b82f6;"></div></div>
          <span class="hbar-pct">${fmtNum(m.out)}</span>
        </div>
        <div class="hbar-row" style="grid-template-columns:90px 1fr 85px;margin-top:4px;">
          <span class="hbar-label" style="color:var(--teal);">Nội trú</span>
          <div class="hbar-track"><div class="hbar-val" style="width:${inW}%;background:#0d9488;"></div></div>
          <span class="hbar-pct">${fmtNum(m.inp)}</span>
        </div>
      </div>`;
  });
  chartHtml += `</div>`;
  document.getElementById("part3EntityChart").innerHTML = chartHtml;

  document.getElementById("part3SummaryTable").innerHTML = `
    <table>
      <thead>
        <tr><th>Thuộc tính quy mô</th><th class="num-cell">Ngoại trú</th><th class="num-cell">Nội trú</th></tr>
      </thead>
      <tbody>
        <tr><td>Tên file CSV</td><td><code>...Trân (1).csv</code></td><td><code>...Trân.csv (1).csv</code></td></tr>
        <tr><td>Số dòng (Rows)</td><td class="num-cell"><strong>${fmtNum(out.rows)}</strong></td><td class="num-cell"><strong>${fmtNum(inp.rows)}</strong></td></tr>
        <tr><td>Số cột (Columns)</td><td class="num-cell">${out.cols}</td><td class="num-cell">${inp.cols}</td></tr>
        <tr><td>Số BeneID duy nhất</td><td class="num-cell">${fmtNum(out.unique.bene)}</td><td class="num-cell">${fmtNum(inp.unique.bene)}</td></tr>
        <tr><td>Số ClaimID duy nhất (hợp lệ)</td><td class="num-cell">${fmtNum(out.unique.claim)}</td><td class="num-cell">${fmtNum(inp.unique.claim)}</td></tr>
        <tr><td>Số Provider duy nhất</td><td class="num-cell">${fmtNum(out.unique.provider)}</td><td class="num-cell">${fmtNum(inp.unique.provider)}</td></tr>
        <tr><td>Số AttendingPhysician duy nhất</td><td class="num-cell">${fmtNum(out.unique.attending)}</td><td class="num-cell">${fmtNum(inp.unique.attending)}</td></tr>
        <tr><td>Số OperatingPhysician duy nhất</td><td class="num-cell">${fmtNum(out.unique.operating)}</td><td class="num-cell">${fmtNum(inp.unique.operating)}</td></tr>
      </tbody>
    </table>
  `;
}

/* ==================== PART 4 ==================== */
let currentAuditMode = "outpatient";
let auditSearchQuery = "";
let auditSeverity = "all";
let auditSortKey = "pct";
let auditSortAsc = false;

function switchAuditDataset(mode) {
  currentAuditMode = mode;
  document.getElementById("btnAuditOut").classList.toggle("active", mode === "outpatient");
  document.getElementById("btnAuditIn").classList.toggle("active", mode === "inpatient");
  document.getElementById("btnAuditCompare").classList.toggle("active", mode === "compare");
  const btnImg = document.getElementById("btnAuditImages");
  if (btnImg) {
    if (mode === "images") {
      btnImg.style.background = "#312e81";
      btnImg.style.boxShadow = "0 0 0 3px rgba(99, 102, 241, 0.4)";
    } else {
      btnImg.style.background = "#4f46e5";
      btnImg.style.boxShadow = "none";
    }
  }
  renderPart4();
}

function renderPart4() {
  const container = document.getElementById("auditDynamicContent");
  const imgs = REPORT_DATA.notebook_images || [];

  if (currentAuditMode === "images") {
    container.innerHTML = `
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">📷 Bộ Sưu Tập 2 Biểu Đồ Hình Ảnh Gốc (Trích xuất trực tiếp từ Notebook)</div>
            <div class="card-subtitle">Sinh bởi thư viện Matplotlib &amp; Seaborn trong cell code số 8 và số 9 của file Bao_Cao_Tieu_Luan_Nhom6.ipynb</div>
          </div>
        </div>

        <div style="margin-bottom:24px;background:#f8fafc;padding:16px;border-radius:12px;border:1px solid var(--border);">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <h4 style="font-size:15px;color:var(--text);">1. Biểu đồ Tỷ lệ thiếu dữ liệu của Top 15 cột thiếu nhiều nhất (Cell 8)</h4>
            <span class="badge badge-blue">Figure 1 • Barplot</span>
          </div>
          <p style="font-size:13px;color:var(--muted);margin-bottom:12px;">
            So sánh trực quan giữa <strong>Ngoại trú (đỏ)</strong> và <strong>Nội trú (xanh dương)</strong>. Cho thấy các biến thủ thuật và chẩn đoán bổ sung có tỷ lệ thiếu rất cao (&gt;70%), trong khi các biến nhân khẩu/bệnh án cốt lõi có độ hoàn thiện tốt hơn.
          </p>
          <div class="nb-img-wrap">
            <img src="data:image/png;base64,${imgs[0] ? imgs[0].b64 : ''}" alt="Biểu đồ tỷ lệ thiếu dữ liệu" style="box-shadow:0 4px 14px rgba(0,0,0,0.08);" />
          </div>
        </div>

        <div style="background:#f8fafc;padding:16px;border-radius:12px;border:1px solid var(--border);">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <h4 style="font-size:15px;color:var(--text);">2. Biểu đồ Phân phối chi trả và Boxplot phát hiện Outliers (Cell 9)</h4>
            <span class="badge badge-rose">Figure 2 • Boxplot</span>
          </div>
          <p style="font-size:13px;color:var(--muted);margin-bottom:12px;">
            Vẽ phân phối số tiền chi trả bồi thường <code>InscClaimAmtReimbursed</code> (&ge; 0). Nhận diện hàng loạt điểm ngoại lai vượt xa ngưỡng $Q_3 + 1.5 \times IQR$ ($690 ở Ngoại trú và $24,000 ở Nội trú), với các khoản chi trả cá biệt lên đến $10.3M và $47.0M.
          </p>
          <div class="nb-img-wrap">
            <img src="data:image/png;base64,${imgs[1] ? imgs[1].b64 : ''}" alt="Biểu đồ Boxplot tài chính" style="box-shadow:0 4px 14px rgba(0,0,0,0.08);" />
          </div>
        </div>
      </div>
    `;
    return;
  }

  if (currentAuditMode === "compare") {
    const out = REPORT_DATA.outpatient;
    const inp = REPORT_DATA.inpatient;

    let imgsHtml = "";
    if (imgs.length > 0) {
      imgsHtml = `
        <div style="margin-top:20px;">
          <h4 style="font-size:15px;margin-bottom:12px;">Hình ảnh biểu đồ gốc đính kèm từ Notebook:</h4>
          ${imgs.map((im, idx) => `
            <div class="nb-img-wrap" style="margin-bottom:16px;">
              <div style="font-weight:700;font-size:13px;margin-bottom:8px;text-align:left;color:var(--muted);">
                Hình ${idx + 1}: Biểu đồ từ cell output [${im.cell_index}]
              </div>
              <img src="data:image/png;base64,${im.b64}" alt="Biểu đồ kiểm toán ${idx + 1}" />
            </div>
          `).join("")}
        </div>
      `;
    }

    container.innerHTML = `
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">So sánh kiểm toán &amp; Phân phối tài chính giữa Ngoại trú và Nội trú</div>
            <div class="card-subtitle">Đối chiếu toàn diện các chỉ số độ sạch và ngoại lai tài chính</div>
          </div>
        </div>
        <div class="table-wrap" style="margin-bottom:18px;">
          <table>
            <thead>
              <tr><th>Tên cột</th><th class="num-cell">Thiếu Ngoại trú (%)</th><th class="num-cell">Thiếu Nội trú (%)</th><th class="num-cell">Số âm Ngoại trú</th><th class="num-cell">Số âm Nội trú</th></tr>
            </thead>
            <tbody>
              ${buildCompareRows(out, inp)}
            </tbody>
          </table>
        </div>
        ${imgsHtml}
      </div>
    `;
    return;
  }

  const ds = REPORT_DATA[currentAuditMode];
  const isOut = currentAuditMode === "outpatient";
  const accent = isOut ? "#e11d48" : "#2563eb";
  const accentSoft = isOut ? "#fff1f2" : "#eff6ff";

  const top15 = [...ds.audit].sort((a, b) => b.pct - a.pct).slice(0, 15);
  const maxBucket = Math.max(...ds.financial.buckets.map(b => b.count), 1);

  container.innerHTML = `
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Tổng số dòng / biến</div>
        <div class="kpi-value">${fmtNum(ds.rows)}</div>
        <div class="kpi-sub">${ds.cols} cột thuộc tính (${ds.name})</div>
      </div>
      <div class="kpi-card rose">
        <div class="kpi-label">Giá trị âm (InscClaimAmtReimbursed)</div>
        <div class="kpi-value">${fmtNum(ds.financial.neg_count)}</div>
        <div class="kpi-sub">Chiếm ${ds.financial.neg_pct}% • Deductible âm: ${fmtNum(ds.financial.deductible_neg)}</div>
      </div>
      <div class="kpi-card teal">
        <div class="kpi-label">Mean / Median Chi trả</div>
        <div class="kpi-value">${fmtCompactUSD(ds.financial.mean)}</div>
        <div class="kpi-sub">Trung vị (Median): ${fmtUSD(ds.financial.median)} | Q3: ${fmtUSD(ds.financial.q3)}</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-label">Outliers (&gt; ${fmtUSD(ds.financial.upper_bound)})</div>
        <div class="kpi-value">${fmtNum(ds.financial.outlier_count)}</div>
        <div class="kpi-sub">Chiếm ${ds.financial.outlier_pct}% hồ sơ hợp lệ • Max: ${fmtCompactUSD(ds.financial.max)}</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Top 15 Cột có Tỷ lệ Thiếu Dữ Liệu Cao nhất</div>
            <div class="card-subtitle">Bao gồm ô trống, NaN, 'None', 'N/A', '?' và '-'</div>
          </div>
        </div>
        <div class="hbar-list">
          ${top15.map(r => `
            <div class="hbar-row" title="Tổng thiếu: ${fmtNum(r.total_miss)} | Blank/NaN: ${fmtNum(r.blank_nan)} | None: ${fmtNum(r.none)} | N/A: ${fmtNum(r.na)} | ?: ${fmtNum(r.qm)} | -: ${fmtNum(r.dash)}">
              <span class="hbar-label">${r.col}</span>
              <div class="hbar-track">
                <div class="hbar-val" style="width:${r.pct}%;background:${accent};"></div>
              </div>
              <span class="hbar-pct">${r.pct}%</span>
            </div>
          `).join("")}
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Phân phối Số tiền Chi trả Hợp lệ (USD)</div>
            <div class="card-subtitle">Phân bố lệch phải mạnh với đuôi ngoại lai cực lớn (Max: ${fmtUSD(ds.financial.max)})</div>
          </div>
        </div>
        <div class="vchart-grid">
          ${ds.financial.buckets.map(b => {
            const hPct = Math.max(4, Math.round((b.count / maxBucket) * 100));
            return `
              <div class="vbar-col">
                <div class="vbar-val-lbl">${fmtNum(b.count)}</div>
                <div class="vbar" style="height:${hPct}%;background:${accent};"></div>
                <div class="vbar-xlbl">${b.label}</div>
              </div>`;
          }).join("")}
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:14px;font-size:12.5px;color:var(--muted);">
          <span><strong>Q1 (25%):</strong> ${fmtUSD(ds.financial.q1)}</span>
          <span><strong>Median (50%):</strong> ${fmtUSD(ds.financial.median)}</span>
          <span><strong>Q3 (75%):</strong> ${fmtUSD(ds.financial.q3)}</span>
          <span><strong>P95:</strong> ${fmtUSD(ds.financial.p95)}</span>
          <span><strong>P99:</strong> ${fmtUSD(ds.financial.p99)}</span>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">Bảng Kiểm toán Chi tiết Từng Cột — ${ds.name}</div>
          <div class="card-subtitle">Bấm vào tiêu đề cột để sắp xếp • Lọc theo ngưỡng dữ liệu thiếu</div>
        </div>
      </div>
      <div class="controls-bar">
        <input type="text" class="search-input" value="${auditSearchQuery}" placeholder="Tìm tên cột (vd: BeneID, Claim, Diagnosis)..." oninput="auditSearchQuery=this.value;renderAuditTableOnly()" />
        <div class="filter-group" id="auditSevFilters">
          <button class="pill-btn ${auditSeverity==='all'?'active':''}" onclick="setAuditSev('all', this)">Tất cả (${ds.cols})</button>
          <button class="pill-btn ${auditSeverity==='low'?'active':''}" onclick="setAuditSev('low', this)">Thấp (&lt;5%)</button>
          <button class="pill-btn ${auditSeverity==='med'?'active':''}" onclick="setAuditSev('med', this)">Trung bình (5–20%)</button>
          <button class="pill-btn ${auditSeverity==='high'?'active':''}" onclick="setAuditSev('high', this)">Cao (20–70%)</button>
          <button class="pill-btn ${auditSeverity==='vhigh'?'active':''}" onclick="setAuditSev('vhigh', this)">Rất cao (&gt;70%)</button>
          <button class="pill-btn ${auditSeverity==='neg'?'active':''}" onclick="setAuditSev('neg', this)">Có số âm (&lt;0)</button>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th onclick="sortAudit('col')">Tên Cột</th>
              <th class="num-cell" onclick="sortAudit('total_miss')">Tổng Thiếu</th>
              <th onclick="sortAudit('pct')">Tỷ Lệ Thiếu (%)</th>
              <th class="num-cell" onclick="sortAudit('blank_nan')">Ô Trống / NaN</th>
              <th class="num-cell" onclick="sortAudit('none')">Chữ None</th>
              <th class="num-cell" onclick="sortAudit('na')">Chữ N/A</th>
              <th class="num-cell" onclick="sortAudit('qm')">Ký tự ?</th>
              <th class="num-cell" onclick="sortAudit('dash')">Ký tự -</th>
              <th class="num-cell" onclick="sortAudit('neg')">Số Âm (&lt;0)</th>
            </tr>
          </thead>
          <tbody id="auditTableBody"></tbody>
        </table>
      </div>
    </div>
  `;
  renderAuditTableOnly();
}

function buildCompareRows(out, inp) {
  const inMap = Object.fromEntries(inp.audit.map(r => [r.col, r]));
  const outMap = Object.fromEntries(out.audit.map(r => [r.col, r]));
  const allCols = [...new Set([...out.audit.map(r => r.col), ...inp.audit.map(r => r.col)])];
  return allCols.map(c => {
    const o = outMap[c];
    const ip = inMap[c];
    return `
      <tr>
        <td><code>${c}</code></td>
        <td class="num-cell">${o ? o.pct + '%' : '<span class="badge badge-slate">Không có</span>'}</td>
        <td class="num-cell">${ip ? ip.pct + '%' : '<span class="badge badge-slate">Không có</span>'}</td>
        <td class="num-cell" style="color:${o && o.neg > 0 ? 'var(--rose)' : 'inherit'};font-weight:${o && o.neg > 0 ? '700' : '400'};">${o ? fmtNum(o.neg) : '-'}</td>
        <td class="num-cell" style="color:${ip && ip.neg > 0 ? 'var(--rose)' : 'inherit'};font-weight:${ip && ip.neg > 0 ? '700' : '400'};">${ip ? fmtNum(ip.neg) : '-'}</td>
      </tr>`;
  }).join("");
}

function setAuditSev(sev, el) {
  auditSeverity = sev;
  document.querySelectorAll("#auditSevFilters .pill-btn").forEach(b => b.classList.remove("active"));
  el.classList.add("active");
  renderAuditTableOnly();
}

function sortAudit(key) {
  if (auditSortKey === key) auditSortAsc = !auditSortAsc;
  else { auditSortKey = key; auditSortAsc = (key === "col"); }
  renderAuditTableOnly();
}

function renderAuditTableOnly() {
  const ds = REPORT_DATA[currentAuditMode];
  if (!ds) return;
  const q = (auditSearchQuery || "").trim().toLowerCase();
  let rows = ds.audit.filter(r => {
    if (q && !r.col.toLowerCase().includes(q)) return false;
    if (auditSeverity === "low") return r.pct < 5;
    if (auditSeverity === "med") return r.pct >= 5 && r.pct < 20;
    if (auditSeverity === "high") return r.pct >= 20 && r.pct <= 70;
    if (auditSeverity === "vhigh") return r.pct > 70;
    if (auditSeverity === "neg") return r.neg > 0;
    return true;
  });

  rows.sort((a, b) => {
    let va = a[auditSortKey], vb = b[auditSortKey];
    if (typeof va === "string") return auditSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return auditSortAsc ? va - vb : vb - va;
  });

  document.getElementById("auditTableBody").innerHTML = rows.map(r => {
    let barColor = "#0d9488";
    if (r.pct > 70) barColor = "#e11d48";
    else if (r.pct >= 20) barColor = "#d97706";
    else if (r.pct >= 5) barColor = "#2563eb";
    return `
      <tr>
        <td><code>${r.col}</code></td>
        <td class="num-cell"><strong>${fmtNum(r.total_miss)}</strong></td>
        <td>
          <div class="bar-track"><div class="bar-fill" style="width:${r.pct}%;background:${barColor};"></div></div>
          <strong>${r.pct}%</strong>
        </td>
        <td class="num-cell">${fmtNum(r.blank_nan)}</td>
        <td class="num-cell">${fmtNum(r.none)}</td>
        <td class="num-cell">${fmtNum(r.na)}</td>
        <td class="num-cell">${fmtNum(r.qm)}</td>
        <td class="num-cell">${fmtNum(r.dash)}</td>
        <td class="num-cell" style="color:${r.neg > 0 ? 'var(--rose)' : 'var(--muted)'};font-weight:${r.neg > 0 ? '700' : '400'};">${fmtNum(r.neg)}</td>
      </tr>`;
  }).join("");
}

/* ==================== PART 5 ==================== */
let activeQ = "q1";
function switchQTab(q, el) {
  activeQ = q;
  document.querySelectorAll("#sec-part5 .subtab-btn").forEach(b => b.classList.remove("active"));
  if (el) el.classList.add("active");
  renderPart5();
}

function renderPart5() {
  const out = REPORT_DATA.outpatient;
  const inp = REPORT_DATA.inpatient;
  const box = document.getElementById("qContent");

  if (activeQ === "q1") {
    const maxOut = Math.max(...out.top_providers.map(p => p.total), 1);
    const maxIn = Math.max(...inp.top_providers.map(p => p.total), 1);
    box.innerHTML = `
      <div class="callout">
        <strong>Câu hỏi 1: Nhận diện các Cơ sở Y tế (Provider) có dấu hiệu trục lợi chi phí bồi thường bất thường</strong>
        <ul style="margin:8px 0 0 18px;">
          <li><strong>Vấn đề thực tế:</strong> Những cơ sở y tế nào nhận dòng tiền bồi thường vượt trội phi lý so với quy mô số lượng bệnh nhân?</li>
          <li><strong>Phương pháp Python:</strong> Group by theo <code>Provider</code> (Tổng lượt claim, Tổng tiền <code>Sum</code>, Trung bình <code>Mean</code>, <code>Std</code>), tính Cost Anomaly Score bằng Z-Score / K-Means / IQR để định vị Top 1% Provider rủi ro.</li>
          <li><strong>Giá trị kinh doanh:</strong> Thu hẹp phạm vi thanh tra, tập trung kiểm toán các cơ sở có nguy cơ gian lận cao nhất.</li>
        </ul>
      </div>
      <div class="grid-2">
        <div>
          <h4 style="margin-bottom:10px;">Top 10 Provider nhận bồi thường cao nhất — Ngoại trú (Sơ bộ từ CSV)</h4>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Provider</th><th class="num-cell">Số Claims</th><th class="num-cell">Tổng chi trả</th><th class="num-cell">Trung bình/Claim</th></tr></thead>
              <tbody>
                ${out.top_providers.map(p => `
                  <tr>
                    <td><code>${p.provider}</code></td>
                    <td class="num-cell">${fmtNum(p.claims)}</td>
                    <td class="num-cell"><strong>${fmtCompactUSD(p.total)}</strong></td>
                    <td class="num-cell">${fmtUSD(p.mean)}</td>
                  </tr>`).join("")}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h4 style="margin-bottom:10px;">Top 10 Provider nhận bồi thường cao nhất — Nội trú (Sơ bộ từ CSV)</h4>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Provider</th><th class="num-cell">Số Claims</th><th class="num-cell">Tổng chi trả</th><th class="num-cell">Trung bình/Claim</th></tr></thead>
              <tbody>
                ${inp.top_providers.map(p => `
                  <tr>
                    <td><code>${p.provider}</code></td>
                    <td class="num-cell">${fmtNum(p.claims)}</td>
                    <td class="num-cell"><strong>${fmtCompactUSD(p.total)}</strong></td>
                    <td class="num-cell">${fmtUSD(p.mean)}</td>
                  </tr>`).join("")}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;
  } else if (activeQ === "q2") {
    const los = inp.los_summary;
    const maxLos = Math.max(...inp.los_buckets.map(b => b.count), 1);
    box.innerHTML = `
      <div class="callout teal">
        <strong>Câu hỏi 2: Tương quan Thời gian nằm viện (Length of Stay), Độ phức tạp bệnh lý và Chi phí nội trú</strong>
        <ul style="margin:8px 0 0 18px;">
          <li><strong>Vấn đề thực tế:</strong> Có hiện tượng cố tình "giữ chân" bệnh nhân nằm viện kéo dài ngày (<code>AdmissionDt</code> → <code>DischargeDt</code>) để rút ruột quỹ BHYT không?</li>
          <li><strong>Phương pháp Python:</strong> Tạo đặc trưng <code>LengthOfStay</code>, <code>Diagnosis_Count</code> (1..10), <code>Procedure_Count</code> (1..6); vẽ Scatter plot, Ma trận tương quan và Hồi quy tuyến tính tìm ca nằm lâu bất thường dù ít bệnh/thủ thuật.</li>
          <li><strong>Giá trị kinh doanh:</strong> Ngăn chặn trục lợi giường bệnh và lạm dụng chỉ định nội trú không cần thiết.</li>
        </ul>
      </div>
      <div class="kpi-grid">
        <div class="kpi-card teal"><div class="kpi-label">Số ca tính được LOS hợp lệ</div><div class="kpi-value">${fmtNum(los.valid_count)}</div><div class="kpi-sub">Trên tổng ${fmtNum(inp.rows)} hồ sơ Nội trú</div></div>
        <div class="kpi-card"><div class="kpi-label">Số ngày nằm viện TB</div><div class="kpi-value">${los.mean_days} ngày</div><div class="kpi-sub">Trung vị: ${los.median_days} ngày | Max: ${los.max_days} ngày</div></div>
        <div class="kpi-card violet"><div class="kpi-label">Số mã chẩn đoán TB / ca</div><div class="kpi-value">${los.avg_diag_count}</div><div class="kpi-sub">Đếm từ ClmDiagnosisCode_1..10</div></div>
        <div class="kpi-card amber"><div class="kpi-label">Số thủ thuật TB / ca</div><div class="kpi-value">${los.avg_proc_count}</div><div class="kpi-sub">Đếm từ ClmProcedureCode_1..6</div></div>
      </div>
      <div class="card" style="box-shadow:none;background:#f8fafc;">
        <div class="card-title" style="margin-bottom:10px;">Phân phối Thời gian nằm viện Nội trú (Length of Stay — Dữ liệu thực tế CSV)</div>
        <div class="vchart-grid">
          ${inp.los_buckets.map(b => {
            const hPct = Math.max(4, Math.round((b.count / maxLos) * 100));
            return `
              <div class="vbar-col">
                <div class="vbar-val-lbl">${fmtNum(b.count)}</div>
                <div class="vbar" style="height:${hPct}%;background:var(--teal);"></div>
                <div class="vbar-xlbl">${b.label}</div>
              </div>`;
          }).join("")}
        </div>
      </div>
    `;
  } else {
    box.innerHTML = `
      <div class="callout amber">
        <strong>Câu hỏi 3: Phát hiện hành vi bất thường và mâu thuẫn chuyên môn của Đội ngũ Y bác sĩ</strong>
        <ul style="margin:8px 0 0 18px;">
          <li><strong>Vấn đề thực tế:</strong> Có tồn tại bác sĩ "ma" hoặc một bác sĩ vừa điều trị chính (<code>AttendingPhysician</code>) vừa tự phẫu thuật (<code>OperatingPhysician</code>) với tần suất dày đặc tại nhiều cơ sở khác nhau không?</li>
          <li><strong>Phương pháp Python:</strong> Lập hồ sơ bác sĩ (số ca mổ tối đa/ngày), tính tỷ lệ trùng <code>AttendingPhysician == OperatingPhysician</code> theo Provider, và Phân tích mạng lưới liên kết (Network Analysis).</li>
          <li><strong>Giá trị kinh doanh:</strong> Phát hiện gian lận có tổ chức giữa cơ sở y tế và nhân sự bác sĩ.</li>
        </ul>
      </div>
      <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">Ngoại trú: Attending == Operating</div><div class="kpi-value">${out.physician_stats.same_pct}%</div><div class="kpi-sub">${fmtNum(out.physician_stats.same_attending_operating)} / ${fmtNum(out.physician_stats.both_valid_count)} ca có đủ 2 mã BS</div></div>
        <div class="kpi-card rose"><div class="kpi-label">Nội trú: Attending == Operating</div><div class="kpi-value">${inp.physician_stats.same_pct}%</div><div class="kpi-sub">${fmtNum(inp.physician_stats.same_attending_operating)} / ${fmtNum(inp.physician_stats.both_valid_count)} ca có đủ 2 mã BS</div></div>
      </div>
      <div class="grid-2">
        <div>
          <h4 style="margin-bottom:10px;">Top Bác sĩ Phẫu thuật nhiều hồ sơ nhất — Ngoại trú</h4>
          <table>
            <thead><tr><th>Mã OperatingPhysician</th><th class="num-cell">Số lượt Claim</th></tr></thead>
            <tbody>
              ${out.physician_stats.top_operating.map(p => `<tr><td><code>${p.physician}</code></td><td class="num-cell"><strong>${fmtNum(p.claims)}</strong></td></tr>`).join("")}
            </tbody>
          </table>
        </div>
        <div>
          <h4 style="margin-bottom:10px;">Top Bác sĩ Phẫu thuật nhiều hồ sơ nhất — Nội trú</h4>
          <table>
            <thead><tr><th>Mã OperatingPhysician</th><th class="num-cell">Số lượt Claim</th></tr></thead>
            <tbody>
              ${inp.physician_stats.top_operating.map(p => `<tr><td><code>${p.physician}</code></td><td class="num-cell"><strong>${fmtNum(p.claims)}</strong></td></tr>`).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }
}

// Khởi chạy dashboard
renderOverview();
renderPart1();
initPart2();
renderPart3();
renderPart4();
renderPart5();
</script>
</body>
</html>
"""


def main():
    print("Đang đọc và tính toán dữ liệu Ngoại trú...")
    out_stats = compute_dataset_stats(F1_PATH, "Ngoại Trú (Outpatient)")
    print("Đang đọc và tính toán dữ liệu Nội trú...")
    in_stats = compute_dataset_stats(F2_PATH, "Nội Trú (Inpatient)")
    print("Đang trích xuất biểu đồ gốc từ Notebook...")
    nb_images = extract_notebook_images(NB_PATH)

    dictionary = [
        {
            "stt": r[0], "name": r[1], "meaning": r[2], "unit": r[3],
            "dtype": r[4], "table": r[5], "role": r[6], "note": r[7]
        }
        for r in DATA_DICTIONARY
    ]

    payload = {
        "outpatient": out_stats,
        "inpatient": in_stats,
        "dictionary": dictionary,
        "notebook_images": nb_images,
    }

    html = HTML_TEMPLATE.replace("__REPORT_DATA_JSON__", json.dumps(payload, ensure_ascii=False))
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Đã tạo dashboard thành công: {OUT_HTML} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()

