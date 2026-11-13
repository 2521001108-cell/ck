import base64
import os
import subprocess

def create_preview_html():
    work_dir = r"c:\Users\Admin\Downloads\Aa"
    logo_path = os.path.join(work_dir, "Ảnh1.png")
    
    logo_base64 = ""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode("utf-8")
            logo_src = f"data:image/png;base64,{logo_base64}"
    else:
        logo_src = "Ảnh1.png"

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Tiểu Luận - Nhóm 6 - UFM</title>
<style>
@page {{
  size: A4 portrait;
  margin-top: 20mm;
  margin-bottom: 20mm;
  margin-left: 30mm;
  margin-right: 20mm;
}}

body {{
  font-family: "Times New Roman", Times, "Liberation Serif", serif;
  font-size: 13pt;
  line-height: 1.4;
  color: #111111;
  margin: 0;
  padding: 0;
}}

p {{
  margin-top: 0pt;
  margin-bottom: 6pt;
  text-align: justify;
  text-indent: 0.8cm;
}}

.no-indent {{
  text-indent: 0 !important;
}}

h1.section-title {{
  font-size: 16pt;
  font-weight: bold;
  text-align: center;
  text-transform: uppercase;
  margin-top: 20pt;
  margin-bottom: 20pt;
  text-indent: 0 !important;
}}

/* --- TRANG BÌA TIỂU LUẬN (COVER PAGE) --- */
.cover-wrapper {{
  box-sizing: border-box;
  width: 100%;
  height: 250mm; /* Chiều cao in chuẩn A4 trừ lề */
  border: 4px double #111111;
  padding: 24mm 8mm 18mm 8mm;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: center;
  text-align: center;
  background-color: #ffffff;
}}

.cover-section-top,
.cover-section-mid,
.cover-section-bottom {{
  width: 100%;
  text-align: center;
}}

.cover-line {{
  margin-top: 0 !important;
  margin-bottom: 6pt !important;
  text-indent: 0 !important;
  text-align: center !important;
}}

.cover-school {{
  font-size: 14.5pt;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}

.cover-faculty {{
  font-size: 13.5pt;
  font-weight: bold;
  text-transform: uppercase;
}}

.cover-logo-container {{
  margin: 12pt auto 10pt auto;
  text-align: center;
}}

.cover-logo {{
  width: 90px;
  height: auto;
  display: block;
  margin: 0 auto;
}}

.cover-divider {{
  width: 130px;
  height: 1.5px;
  background-color: #111111;
  margin: 8pt auto 18pt auto;
}}

.cover-type {{
  font-size: 20pt;
  font-weight: bold;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 12pt !important;
}}

.cover-subject {{
  font-size: 12.5pt;
  font-weight: bold;
  text-transform: uppercase;
  white-space: nowrap;
  margin-bottom: 24pt !important;
}}

.cover-topic-label {{
  font-size: 13.5pt;
  font-weight: bold;
  text-transform: uppercase;
  margin-bottom: 8pt !important;
}}

.cover-topic-title {{
  font-size: 15.5pt;
  font-weight: bold;
  text-transform: uppercase;
  line-height: 1.45;
  color: #0b2545;
  margin-bottom: 25pt !important;
  padding: 0 5px;
}}

.cover-instructor,
.cover-class,
.cover-group,
.cover-student {{
  font-size: 13pt;
  font-weight: bold;
  margin-bottom: 6pt !important;
}}

.cover-location {{
  font-size: 12pt;
  font-weight: bold;
  text-transform: uppercase;
  margin-top: 25pt !important;
  margin-bottom: 0 !important;
}}

.page-break {{
  page-break-after: always;
  break-after: page;
}}

.signature-block {{
  text-align: right;
  margin-top: 36pt;
  text-indent: 0 !important;
}}

.signature-block p {{
  text-indent: 0 !important;
  text-align: right;
  margin-bottom: 4pt;
}}

.signature-space {{
  height: 50pt;
}}
</style>
</head>
<body>

<!-- TRANG 1: TRANG BÌA -->
<div class="cover-wrapper">
  <div class="cover-section-top">
    <p class="cover-line cover-school">TRƯỜNG ĐẠI HỌC TÀI CHÍNH - MARKETING</p>
    <p class="cover-line cover-faculty">KHOA KHOA HỌC DỮ LIỆU</p>
    <div class="cover-logo-container">
      <img src="{logo_src}" alt="Logo UFM" class="cover-logo">
    </div>
    <div class="cover-divider"></div>
  </div>

  <div class="cover-section-mid">
    <p class="cover-line cover-type">TIỂU LUẬN</p>
    <p class="cover-line cover-subject">HỌC PHẦN: LẬP TRÌNH PYTHON CHO KHOA HỌC DỮ LIỆU</p>
    <p class="cover-line cover-topic-label">ĐỀ TÀI</p>
    <p class="cover-line cover-topic-title">XÂY DỰNG MÔ HÌNH DỰ BÁO CHI PHÍ BỒI THƯỜNG BẢO HIỂM Y TẾ</p>
  </div>

  <div class="cover-section-bottom">
    <p class="cover-line cover-instructor">Giảng viên: ThS. Trần Mạnh Tường</p>
    <p class="cover-line cover-class">Lớp: 2631101190501</p>
    <p class="cover-line cover-group">Nhóm thực hiện: Nhóm 6</p>
    <p class="cover-line cover-student">Sinh viên thực hiện: Phan Thị Quế Trân (MSSV: 2521001108)</p>
    <p class="cover-line cover-location">TP. HỒ CHÍ MINH, NĂM 2026</p>
  </div>
</div>

<div class="page-break"></div>

<!-- TRANG 2: LỜI CẢM ƠN -->
<div>
  <h1 class="section-title">LỜI CẢM ƠN</h1>
  
  <p>Lời đầu tiên, nhóm chúng em xin gửi lời cảm ơn đến quý thầy cô trường Đại học Tài chính - Marketing nói chung và quý thầy cô khoa Khoa học Dữ liệu nói riêng về sự chỉ dẫn tận tình và tạo điều kiện thuận lợi cho sinh viên chúng em được tiếp nhận nhiều bài học quý giá cũng như những trải nghiệm thực tế quý báu.</p>
  
  <p>Đặc biệt hơn hết, chúng em xin gửi lời cảm ơn chân thành và sâu sắc nhất đến thầy ThS. Trần Mạnh Tường, đã tận tình hỗ trợ nhóm trong khoảng thời gian thực hiện nghiên cứu này. Chúng em đã nhận được sự hướng dẫn rất chi tiết cùng những lời khuyên của thầy từ lúc chọn đề tài, cho đến lúc góp ý về mặt nội dung cũng như hình thức để có thể hoàn thành bài một cách tốt nhất.</p>
  
  <p>Trong quá trình thực hiện đề tài nghiên cứu, nhóm không tránh khỏi những sai sót về nội dung cũng như hình thức nhất định, chúng em rất mong sẽ nhận được những góp ý từ thầy để có thể hoàn thiện hơn và có thêm những bài học quý báu để làm hành trang cho sự nghiệp sau này.</p>
  
  <p>Lời cuối, chúng em xin kính chúc quý Thầy Cô trong khoa Khoa học Dữ liệu và thầy ThS. Trần Mạnh Tường có thật nhiều sức khỏe, luôn tràn đầy nhiệt huyết và thành công trong sứ mệnh cao quý của mình.</p>
  
  <p>Chúng em xin chân thành cảm ơn!</p>
  
  <div class="signature-block">
    <p><em>Thành phố Hồ Chí Minh, năm 2026</em></p>
    <p><strong>Nhóm tác giả thực hiện</strong></p>
    <p><em>(Ký và ghi rõ họ tên)</em></p>
    <div class="signature-space"></div>
    <p><strong>Nhóm 6</strong></p>
  </div>
</div>

<div class="page-break"></div>

<!-- TRANG 3: LỜI CAM ĐOAN -->
<div>
  <h1 class="section-title">LỜI CAM ĐOAN</h1>
  
  <p>Nhóm sinh viên xin cam đoan đề tài <strong>“Xây dựng Mô hình Dự báo Chi phí Bồi thường Bảo hiểm Y tế”</strong> là một công trình nghiên cứu của nhóm tác giả dưới sự hỗ trợ của giảng viên hướng dẫn là ThS. Trần Mạnh Tường, không có bất cứ sự sao chép nào. Nhóm đã nỗ lực tìm hiểu, khảo sát, nghiên cứu và vận dụng những kiến thức đã tích lũy được trong quá trình học tập tại trường để thực hiện đề tài này.</p>
  
  <p>Các số liệu, kết quả, nội dung trình bày trong nghiên cứu là trung thực, khách quan và có nguồn gốc rõ ràng. Một số tài liệu mà nhóm tham khảo cũng đã có trích nguồn cụ thể ở cuối bài. Nhóm tác giả xin chịu hoàn toàn trách nhiệm trước giảng viên hướng dẫn, khoa và Nhà trường khi có bất kỳ vấn đề nào xảy ra.</p>
  
  <div class="signature-block">
    <p><em>TP. Hồ Chí Minh, ngày ... tháng ... năm 2026</em></p>
    <p><strong>Đại diện nhóm tác giả</strong></p>
    <p><em>(Ký và ghi rõ họ tên)</em></p>
    <div class="signature-space"></div>
    <p><strong>Phan Thị Quế Trân</strong></p>
  </div>
</div>

</body>
</html>
"""
    output_html = os.path.join(work_dir, "2521001108_Nhom6_Tieuluan.html")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML created at: {output_html}")
    return output_html

def render_to_pdf(html_path):
    work_dir = r"c:\Users\Admin\Downloads\Aa"
    output_pdf = os.path.join(work_dir, "2521001108_Nhom6_Tieuluan.pdf")
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    
    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_pdf}",
        f"file:///{html_path.replace(os.sep, '/')}"
    ]
    print("Running Edge headless command...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(output_pdf):
        print(f"PDF successfully rendered! Size: {os.path.getsize(output_pdf)} bytes")
        print(f"File path: {output_pdf}")
    else:
        print("Failed to generate PDF. Stderr:", res.stderr)

if __name__ == "__main__":
    html_file = create_preview_html()
    render_to_pdf(html_file)
