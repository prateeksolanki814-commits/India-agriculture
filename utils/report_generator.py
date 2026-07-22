"""
utils/report_generator.py
============================
Builds downloadable reports (PDF, Excel, CSV) from a filtered slice of
the agriculture dataset, plus a short rule-based "AI Summary" narrative.
Kept separate from the Streamlit page so the report-building logic is
testable on its own and reusable (e.g. by a future scheduled-report
feature) without importing streamlit.
"""

import io
import datetime
import pandas as pd
from fpdf import FPDF


def generate_ai_summary(df: pd.DataFrame, scope_label: str) -> str:
    """
    Produces a short, rule-based natural-language summary of the given
    (already filtered) dataframe. Deterministic and data-grounded rather
    than a generative model, so it never invents figures.
    """
    if df.empty:
        return f"No data available for {scope_label}."

    total_area = df["area_ha"].sum()
    total_production = df["production_tonnes"].sum()
    avg_yield = df["yield_t_per_ha"].mean()
    total_profit = df["profit_inr"].sum()
    top_crop = df.groupby("crop")["production_tonnes"].sum().idxmax()
    avg_rainfall = df["rainfall_mm"].mean()
    avg_drought_risk = df["drought_risk_score"].mean()
    sustainability = df["sustainability_score"].mean()

    lines = [
        f"Summary for {scope_label}:",
        f"- Total cultivated area: {total_area:,.0f} hectares across {df['crop'].nunique()} crop(s) "
        f"and {df['district'].nunique()} district(s).",
        f"- Total production: {total_production:,.0f} tonnes, with an average yield of {avg_yield:.2f} t/ha.",
        f"- {top_crop} is the dominant crop by total production in this selection.",
        f"- Estimated net profit: ₹{total_profit:,.0f}, based on modeled revenue and cost figures.",
        f"- Average rainfall was {avg_rainfall:.0f} mm, with an average drought-risk score of "
        f"{avg_drought_risk:.0f}/100.",
        f"- Average sustainability score across this selection is {sustainability:.0f}/100.",
    ]
    return "\n".join(lines)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame, summary_text: str, sheet_name="Report") -> bytes:
    """Builds a multi-sheet Excel workbook: raw data + a summary sheet."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        summary_df = pd.DataFrame({"AI Summary": summary_text.split("\n")})
        summary_df.to_excel(writer, sheet_name="AI Summary", index=False)

        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#2E7D32", "font_color": "white"})
        worksheet = writer.sheets[sheet_name]
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, header_fmt)
            worksheet.set_column(col_idx, col_idx, max(12, len(col_name) + 2))
    return buffer.getvalue()


class _ReportPDF(FPDF):
    """Thin FPDF subclass adding a consistent header/footer to every page."""

    def __init__(self, title: str):
        super().__init__()
        self.report_title = title

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(46, 125, 50)  # brand green
        self.cell(0, 10, "AGRI VISION AI", ln=1)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, self.report_title, ln=1)
        self.set_draw_color(46, 125, 50)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()} | Generated {datetime.date.today().isoformat()}", align="C")


def _pdf_safe_text(text: str) -> str:
    """Core PDF fonts (Helvetica) don't support the ₹ glyph; swap for 'Rs.' in PDF output only."""
    return text.replace("₹", "Rs. ")


def dataframe_to_pdf_bytes(df: pd.DataFrame, summary_text: str, title: str,
                             max_rows: int = 40) -> bytes:
    """
    Builds a simple, clean PDF report: title, AI summary section, then a
    data table (capped at max_rows to keep the PDF a reasonable size --
    the full data is available via the CSV/Excel exports).
    """
    pdf = _ReportPDF(title)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(27, 46, 31)
    pdf.cell(0, 8, "AI Summary", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    for line in _pdf_safe_text(summary_text).split("\n"):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(190, 6, line)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Data Table (showing up to {max_rows} rows)", ln=1)

    display_cols = [c for c in [
        "year", "state", "district", "crop", "area_ha", "production_tonnes",
        "yield_t_per_ha", "profit_inr",
    ] if c in df.columns]
    table_df = df[display_cols].head(max_rows)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(46, 125, 50)
    pdf.set_text_color(255, 255, 255)
    col_width = 190 / len(display_cols)
    for col in display_cols:
        pdf.cell(col_width, 7, col.replace("_", " ").title(), border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(30, 30, 30)
    for i, (_, row) in enumerate(table_df.iterrows()):
        fill = i % 2 == 0
        pdf.set_fill_color(244, 247, 243)
        for col in display_cols:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:,.1f}"
            pdf.cell(col_width, 6, _pdf_safe_text(str(val)), border=1, fill=fill, align="C")
        pdf.ln()

    return bytes(pdf.output())
