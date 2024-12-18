import argparse
import os
import os
import fnmatch


def list_top_level_folders(directory, include_top_level_dirs, exclude_dirs):
    folders = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir():
                    if entry.name in exclude_dirs:
                        continue
                    if entry.name in include_top_level_dirs:
                        folders.append(entry.name)
                    else:
                        folders.append(f"{entry.name} (skipped)")
    except Exception as e:
        print(f"Error reading directory {directory}: {e}")
    return folders


def collect_files(directory, patterns, include_top_level_dirs, exclude_dirs):
    collected_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # If we are at the top level, filter dirs based on include_top_level_dirs
        if root == directory:
            dirs[:] = [
                d for d in dirs if d in include_top_level_dirs or os.path.join(root, d) in include_top_level_dirs
            ]

        for file in files:
            if any(fnmatch.fnmatch(file, pattern) for pattern in patterns):
                collected_files.append(os.path.join(root, file))
    return collected_files


def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_combined_file(output_file, folders, files):
    with open(output_file, "w", encoding="utf-8") as outfile:
        # Write top-level folder structure
        outfile.write("+top-level project structure\n")
        for folder in folders:
            outfile.write(f"{folder}\n")
        outfile.write("-top-level project structure\n\n")

        # Write collected files content
        for file in files:
            relative_path = os.path.relpath(file)
            content = read_file_content(file)
            outfile.write(f"+{relative_path}\n")
            outfile.write(content)
            outfile.write(f"\n-{relative_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Combine project files for analysis.")
    parser.add_argument("--directory", help="Root directory to start the search", default=".")
    parser.add_argument(
        "--out",
        help="Output file to save combined content",
        default="project.files",
    )
    args = parser.parse_args()

    # extensions = [".py", "*.md"]
    patterns = ["*.py", "*.files*"]

    # exclude_dirs = [".git", "build_dir", ".github", "tests"]
    exclude_dirs = [".git", ".github"]

    # part = ["agents", "base", "conditions", "state", "task", "teams", "ui"]
    part = ["_autogen.projectfiles", "agents", "test_utils", "tests", "tools", "books_sql"]
    include_top_level_dirs = part

    folders = list_top_level_folders(args.directory, include_top_level_dirs, exclude_dirs)

    files = collect_files(
        directory=args.directory,
        patterns=patterns,
        include_top_level_dirs=include_top_level_dirs,
        exclude_dirs=exclude_dirs,
    )

    if os.path.exists(args.out):
        os.remove(args.out)

    write_combined_file(args.out, folders, files)


if __name__ == "__main__":
    main()
