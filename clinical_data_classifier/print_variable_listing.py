"""Prints the variable listings for easy copy-paste into files.
"""
from docx import Document

filepath9202 = "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9202/RTOG 9202 Variable Listing.docx"
filepath9408 = "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9408/RTOG 9408 Variable Listing.docx"
filepath9413 = "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9413/RTOG 9413 Variable Listing updated Feb-13.docx"
filepath9910 = "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 9910/RTOG 9910 Variable Listing 2.docx"
filepath0126 = "/export/medical_ai/ucsf/box_data/Aperio Images of NRG GU H&E Slides/NRG Statistics/RTOG 0126/RTOG 0126 Variable Listing 2.docx"

filepath = filepath0126

document = Document(filepath)
data = []
for table in document.tables:
    for row in table.rows:
        row_data = []
        for cell in row.cells:
            row_data.append(cell.text.encode('utf-8'))
        data.append(row_data)

variables = {row[2] : row[3] for row in data}
def format_var(var):
    if var.lower().strip() in {b'continuous', b'continious'}:
        var = "\'c\'"
    elif var.lower().strip() in {b'text', b'texts', b'text field'}:
        var = "\'s\'"
    else:
        var = b"{" + var + b"}"
    return var

for key, val in variables.items():
    vn = "{}".format(str(key).lower().replace(" ", ""))
    vf = "{}".format(format_var(val))
    print("{} : [{}],".format(vn, vf))
