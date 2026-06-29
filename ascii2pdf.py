from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register a monospace font (Courier is built-in, but you can load others like Consolas/DejaVuSansMono)
# pdfmetrics.registerFont(TTFont('DejaVuSansMono', 'DejaVuSansMono.ttf'))
pdfmetrics.registerFont(TTFont('Menlo', 'Menlo.ttc'))

def ascii_to_pdf(input_file, output_file):
    c = canvas.Canvas(output_file, pagesize=letter)
    # width, height = letter
    width, height = A4
    # print(letter)
    # print(A4)

    # Use monospace font
    # c.setFont("DejaVuSansMono", 10)
    c.setFont("Menlo", 10)

    # Starting position (top margin)
    # x_margin, y_margin = 50, height - 50
    # line_height = 12
    line_height = 10
    x_margin, y_margin = 30, height - 120


    with open(input_file, "r") as f:
        y = y_margin
        for line in f:
            # If page is full, start new page
            if y < 50:
                c.showPage()
                c.setFont("Menlo", 10)
                y = y_margin
            c.drawString(x_margin, y, line.rstrip("\n"))
            y -= line_height

    c.save()

ascii_to_pdf("TRG_ABCD_1.md", "TRG_ABCD_1.pdf")
ascii_to_pdf("TRG_ABCD_2.md", "TRG_ABCD_2.pdf")
ascii_to_pdf("TRG_ABCD_3.md", "TRG_ABCD_3.pdf")
