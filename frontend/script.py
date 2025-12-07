import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def get_code_files(directory, excluded_files=None, excluded_dirs=None):
    """Fetch all TS/TSX project files excluding sensitive configuration files."""
    if excluded_files is None:
        # Added sensitive files to exclusion list
        excluded_files = {
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            ".DS_Store",
            "Thumbs.db",
            "Desktop.ini",
            # Sensitive configuration files
            ".env",
            ".env.development",
            ".env.production",
            ".env.local",
            ".env.staging",
            ".env.test",
            "keys.ts",
            "keys.js",
            "secrets.ts",
            "secrets.js",
            "config.ts",
            "config.js",
            "db.ts",
            "db.js",
            "database.ts",
            "database.js",
            "auth.config.ts",
            "auth.config.js",
            "firebase.config.ts",
            "firebase.config.js",
            "aws.config.ts",
            "aws.config.js",
            "api.config.ts",
            "api.config.js",
            "prisma.config.ts",
            "mongoose.config.ts",
            "supabase.config.ts",
        }

    if excluded_dirs is None:
        excluded_dirs = {
            "node_modules",
            ".git",
            "_pycache_",
            "build",
            "dist",
            ".next",
            "coverage",
            ".nyc_output",
            "logs",
            "uploads",
            "config",
            "secrets",
            ".env.d",
            "prisma/migrations",  # Exclude migration files
            "types/generated",  # Exclude generated types
            ".turbo",  # Exclude Turbo cache
            ".vercel",  # Exclude Vercel config
        }

    code_files = {}

    # Define TS/TSX file extensions we want to include
    ts_extensions = {".ts", ".tsx", ".d.ts", ".prisma"}

    # Define safe configuration files to include (non-sensitive)
    safe_config_files = {
        "package.json",
        "tsconfig.json",
        "tsconfig.build.json",
        "tsconfig.paths.json",
        "tailwind.config.ts",
        "tailwind.config.js",
        "postcss.config.js",
        "postcss.config.ts",
        "webpack.config.ts",
        "webpack.config.js",
        "vite.config.ts",
        "vite.config.js",
        "next.config.js",
        "next.config.ts",
        "babel.config.js",
        "babel.config.ts",
        "eslint.config.js",
        "eslint.config.ts",
        ".eslintrc.js",
        ".eslintrc.ts",
        "prettier.config.js",
        "prettier.config.ts",
        "jest.config.js",
        "jest.config.ts",
        "vitest.config.ts",
        "playwright.config.ts",
        "cypress.config.ts",
    }

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        # Skip if current directory is an excluded directory
        if any(excluded_dir in root.split(os.sep) for excluded_dir in excluded_dirs):
            continue

        for file in files:
            # Skip excluded files
            if file in excluded_files:
                continue

            # Additional check for any file containing sensitive patterns
            if any(
                pattern in file.lower()
                for pattern in [
                    ".env",
                    "secret",
                    "key",
                    "password",
                    "token",
                    "credential",
                ]
            ):
                continue

            file_path = os.path.join(root, file)

            # Get file extension
            _, ext = os.path.splitext(file)

            # Only include TS/TSX files OR safe configuration files
            if ext.lower() in ts_extensions or file in safe_config_files:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code_files[file_path] = f.readlines()

                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")
                    code_files[file_path] = [f"[Error reading file: {str(e)}]"]

    return code_files


def create_pdf(code_data, output_pdf="Code_Export.pdf"):
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    margin = 20 * mm
    line_height = 10
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "📁 TypeScript Project Code Export")
    y -= 2 * line_height
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "📁 TypeScript/TSX Files & Safe Config:")
    y -= 2 * line_height

    file_paths = sorted(list(code_data.keys()))

    # 1. File list with file type indicators
    c.setFont("Courier", 8)
    for path in file_paths:
        if y < margin:
            c.showPage()
            c.setFont("Courier", 8)
            y = height - margin

        display_path = os.path.relpath(path)

        # Add file type indicator
        if display_path.endswith(".ts"):
            file_type = "[TS]"
        elif display_path.endswith(".tsx"):
            file_type = "[TSX]"
        elif display_path.endswith(".d.ts"):
            file_type = "[DTS]"
        elif display_path.endswith("package.json"):
            file_type = "[PKG]"
        elif display_path.endswith("tsconfig.json"):
            file_type = "[TSC]"
        elif display_path.endswith(".config.ts") or display_path.endswith(".config.js"):
            file_type = "[CFG]"
        else:
            file_type = "[CONFIG]"

        c.drawString(margin, y, f"- {file_type} {display_path}")
        y -= line_height

    # Add page break before code content
    c.showPage()
    y = height - margin

    # 2. File contents
    for file_path in file_paths:
        lines = code_data[file_path]
        print(f"📄 Adding: {file_path}")

        if y < margin + 3 * line_height:
            c.showPage()
            y = height - margin

        # File header
        rel_path = os.path.relpath(file_path)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"📄 File: {rel_path}")
        y -= line_height

        # Add separator line
        c.setFont("Courier", 8)
        c.drawString(margin, y, "=" * 80)
        y -= line_height

        # File content with line numbers
        for line_num, line in enumerate(lines, 1):
            if y < margin:
                c.showPage()
                c.setFont("Courier", 8)
                y = height - margin

            # Clean and truncate line
            line = line.strip("\n").encode("latin-1", "replace").decode("latin-1")

            # Add line numbers for all files
            display_line = f"{line_num:3d}: {line[:280]}"

            c.drawString(margin, y, display_line)
            y -= line_height

        # Add spacing between files
        y -= line_height
        if y > margin:
            c.setFont("Courier", 8)
            c.drawString(margin, y, "-" * 80)
            y -= 2 * line_height

    c.save()
    print(f"✅ PDF successfully created: {output_pdf}")
    print(f"📊 Total files processed: {len(code_data)}")
    print(f"📁 File breakdown:")

    # Print file type breakdown
    ts_count = sum(
        1 for f in code_data.keys() if f.endswith(".ts") and not f.endswith(".d.ts")
    )
    tsx_count = sum(1 for f in code_data.keys() if f.endswith(".tsx"))
    dts_count = sum(1 for f in code_data.keys() if f.endswith(".d.ts"))
    config_count = len(code_data) - ts_count - tsx_count - dts_count

    print(f"   - TypeScript files: {ts_count}")
    print(f"   - TSX files: {tsx_count}")
    print(f"   - Type definition files: {dts_count}")
    print(f"   - Safe configuration files: {config_count}")


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Expanded exclusions to include sensitive files
    excluded_files = {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        ".DS_Store",
        "Thumbs.db",
        "Desktop.ini",
        # Sensitive files
        ".env",
        ".env.development",
        ".env.production",
        ".env.local",
        ".env.staging",
        ".env.test",
        "keys.ts",
        "keys.js",
        "secrets.ts",
        "secrets.js",
        "config.ts",
        "config.js",
        "db.ts",
        "db.js",
        "database.ts",
        "database.js",
        "auth.config.ts",
        "auth.config.js",
        "firebase.config.ts",
        "firebase.config.js",
        "aws.config.ts",
        "aws.config.js",
        "api.config.ts",
        "api.config.js",
        "prisma.config.ts",
        "mongoose.config.ts",
        "supabase.config.ts",
    }

    # Directories to exclude (including sensitive directories)
    excluded_dirs = {
        "node_modules",
        ".git",
        "_pycache_",
        "build",
        "dist",
        ".next",
        "coverage",
        ".nyc_output",
        "logs",
        "uploads",
        "config",
        "secrets",
        ".env.d",
        # "prisma/migrations",
        "types/generated",
        ".turbo",
        ".vercel",
    }

    print("🔍 Scanning for TS/TSX files and safe configuration files...")
    print("🔒 Sensitive files (.env, keys, secrets, etc.) will be excluded")

    code_files = get_code_files(root_dir, excluded_files, excluded_dirs)

    if not code_files:
        print("❌ No TS/TSX files found to process!")
        return

    print(f"📁 Found {len(code_files)} files to include in PDF")

    # Show user what files will be included
    print("\n📋 Files to be included:")
    for file_path in sorted(code_files.keys()):
        rel_path = os.path.relpath(file_path)
        print(f"   📄 {rel_path}")

    create_pdf(code_files)

if __name__ == "__main__":
    main()