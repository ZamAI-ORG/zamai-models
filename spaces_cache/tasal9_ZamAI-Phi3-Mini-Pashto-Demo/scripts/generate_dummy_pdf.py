from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# Quick helper to create a dummy Pashto-like PDF for pipeline testing.

def create_pdf(path='data/raw_pdfs/dummy.pdf'):
    c = canvas.Canvas(path, pagesize=A4)
    text = c.beginText(40, 800)
    lines = [
        "دا يو ازمايښتي متن دی.",
        "موږ غواړو چې د پي ډي اف استخراج پروسه وازمويو.",
        "دا بايد د پښتو حروف ولري.",
    ]
    for line in lines:
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    c.save()
    print(f"Created {path}")

if __name__ == '__main__':
    create_pdf()
