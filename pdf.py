import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def get_code_files(directory, excluded_files=None, excluded_dirs=None):
    """Fetch all Python project files, excluding cache + binary files."""

    if excluded_files is None:
        excluded_files = {
            ".DS_Store",
            "Thumbs.db",
            "Desktop.ini",
            "best_model.pth",     # binary
        }

    if excluded_dirs is None:
        excluded_dirs = {
            "__pycache__",
            ".git",
            ".idea",
            ".vscode",
        }

    valid_extensions = {".py", ".txt"}

    code_files = {}

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        for file in files:
            if file in excluded_files:
                continue

            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)

            if ext.lower() not in valid_extensions:
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    code_files[file_path] = f.readlines()

            except Exception as e:
                code_files[file_path] = [f"[Error reading file: {str(e)}]"]

    return code_files


def wrap_text(line, max_chars=100):
    """Wrap long lines into chunks without cutting the PDF."""
    chunks = []
    line = line.rstrip("\n")
    while len(line) > max_chars:
        chunks.append(line[:max_chars])
        line = line[max_chars:]
    chunks.append(line)
    return chunks


def create_pdf(code_data, output_pdf="Backend_Code_Export.pdf"):
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    line_height = 10
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "📁 Backend Project Code Export")
    y -= 2 * line_height

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "📄 Python Files:")
    y -= 2 * line_height

    file_paths = sorted(list(code_data.keys()))

    # List of files
    c.setFont("Courier", 8)
    for path in file_paths:
        if y < margin:
            c.showPage()
            c.setFont("Courier", 8)
            y = height - margin

        display_path = os.path.relpath(path)
        ext = os.path.splitext(path)[1].lower()

        if ext == ".py":
            file_type = "[PY]"
        else:
            file_type = "[TXT]"

        c.drawString(margin, y, f"- {file_type} {display_path}")
        y -= line_height

    # New page for content
    c.showPage()
    y = height - margin

    # Write file contents
    for file_path in file_paths:
        lines = code_data[file_path]
        rel_path = os.path.relpath(file_path)

        if y < margin + 4 * line_height:
            c.showPage()
            y = height - margin

        # File header
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"📄 File: {rel_path}")
        y -= line_height

        c.setFont("Courier", 8)
        c.drawString(margin, y, "=" * 100)
        y -= line_height

        # Content
        for line_num, line in enumerate(lines, start=1):

            wrapped_lines = wrap_text(line, max_chars=110)

            for wrapped in wrapped_lines:
                if y < margin:
                    c.showPage()
                    c.setFont("Courier", 8)
                    y = height - margin

                display_line = f"{line_num:3d}: {wrapped}"
                c.drawString(margin, y, display_line)
                y -= line_height

        # Spacer
        y -= line_height
        if y > margin:
            c.drawString(margin, y, "-" * 100)
            y -= 2 * line_height

    c.save()
    print(f"✅ PDF created: {output_pdf}")
    print(f"📄 Total files included: {len(code_data)}")


def main():
    backend_dir = os.path.join(os.getcwd(), "backend")

    print("🔍 Scanning backend folder...")
    code_files = get_code_files(backend_dir)

    if not code_files:
        print("❌ No backend files found")
        return

    print(f"📁 Files found: {len(code_files)}")
    for f in sorted(code_files.keys()):
        print("   📄", os.path.relpath(f))

    create_pdf(code_files)


if __name__ == "__main__":
    main()
