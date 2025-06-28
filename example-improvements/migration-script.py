#!/usr/bin/env python3
"""
Migration script to modernize scikit-learn codebase with 2025 best practices.

This script automates the application of modern Python features and best practices
to the scikit-learn codebase.

Usage:
    python migration-script.py --dry-run  # Preview changes
    python migration-script.py --apply    # Apply changes
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Generator


class CodeModernizer:
    """Modernizes Python code with 2025 best practices."""
    
    def __init__(self, root_path: Path, dry_run: bool = True) -> None:
        self.root_path = root_path
        self.dry_run = dry_run
        self.changes_made = 0
        
    def find_python_files(self) -> Generator[Path, None, None]:
        """Find all Python files in the project."""
        exclude_patterns = {
            "sklearn/externals",
            "asv_benchmarks",
            "benchmarks",
            "doc/_build",
            "doc/auto_examples",
            "build",
            ".git",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
        }
        
        for py_file in self.root_path.rglob("*.py"):
            # Skip excluded directories
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue
            yield py_file
    
    def add_future_annotations(self, file_path: Path) -> bool:
        """Add 'from __future__ import annotations' to Python files."""
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        
        # Check if already present
        if any("from __future__ import annotations" in line for line in lines[:10]):
            return False
            
        # Find insertion point (after shebang, encoding, and docstring)
        insert_index = 0
        
        # Skip shebang
        if lines and lines[0].startswith("#!"):
            insert_index = 1
            
        # Skip encoding declaration
        if insert_index < len(lines) and "coding" in lines[insert_index]:
            insert_index += 1
            
        # Skip module docstring
        try:
            tree = ast.parse(content)
            if (tree.body and 
                isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Constant) and
                isinstance(tree.body[0].value.value, str)):
                # Find end of docstring
                for i, line in enumerate(lines[insert_index:], insert_index):
                    if '"""' in line or "'''" in line:
                        if line.count('"""') == 2 or line.count("'''") == 2:
                            insert_index = i + 1
                            break
                        elif line.count('"""') == 1 or line.count("'''") == 1:
                            # Multi-line docstring, find closing
                            quote = '"""' if '"""' in line else "'''"
                            for j, next_line in enumerate(lines[i+1:], i+1):
                                if quote in next_line:
                                    insert_index = j + 1
                                    break
                            break
        except SyntaxError:
            pass  # If we can't parse, just insert at the beginning
            
        # Insert the import
        new_lines = lines[:]
        new_lines.insert(insert_index, "from __future__ import annotations")
        new_lines.insert(insert_index + 1, "")
        
        new_content = "\n".join(new_lines)
        
        if not self.dry_run:
            file_path.write_text(new_content, encoding="utf-8")
            
        print(f"{'[DRY RUN] ' if self.dry_run else ''}Added future annotations to {file_path}")
        return True
    
    def standardize_typing_imports(self, file_path: Path) -> bool:
        """Standardize typing imports to use modern patterns."""
        content = file_path.read_text(encoding="utf-8")
        
        # Patterns to modernize
        replacements = [
            # Use collections.abc instead of typing for abstract base classes
            (r"from typing import (.*?)Iterable", r"from collections.abc import Iterable"),
            (r"from typing import (.*?)Mapping", r"from collections.abc import Mapping"),
            (r"from typing import (.*?)Sequence", r"from collections.abc import Sequence"),
            (r"from typing import (.*?)Set", r"from collections.abc import Set"),
            (r"from typing import (.*?)MutableMapping", r"from collections.abc import MutableMapping"),
            
            # Use X | Y syntax for unions (Python 3.10+, but we'll keep Union for now)
            # This would be applied later when minimum Python version is 3.10+
            
            # Group TYPE_CHECKING imports
            (r"from typing import TYPE_CHECKING\n(.*?)from typing import", 
             r"from typing import TYPE_CHECKING, \1"),
        ]
        
        modified = False
        new_content = content
        
        for pattern, replacement in replacements:
            if re.search(pattern, new_content):
                new_content = re.sub(pattern, replacement, new_content)
                modified = True
                
        if modified and not self.dry_run:
            file_path.write_text(new_content, encoding="utf-8")
            
        if modified:
            print(f"{'[DRY RUN] ' if self.dry_run else ''}Standardized typing imports in {file_path}")
            
        return modified
    
    def replace_os_path_with_pathlib(self, file_path: Path) -> bool:
        """Replace os.path usage with pathlib where appropriate."""
        content = file_path.read_text(encoding="utf-8")
        
        # Simple replacements (more complex ones would need AST analysis)
        replacements = [
            (r"import os\.path", "from pathlib import Path"),
            (r"os\.path\.join\((.*?)\)", r"Path(\1)"),
            (r"os\.path\.exists\((.*?)\)", r"Path(\1).exists()"),
            (r"os\.path\.isfile\((.*?)\)", r"Path(\1).is_file()"),
            (r"os\.path\.isdir\((.*?)\)", r"Path(\1).is_dir()"),
            (r"os\.path\.basename\((.*?)\)", r"Path(\1).name"),
            (r"os\.path\.dirname\((.*?)\)", r"str(Path(\1).parent)"),
        ]
        
        modified = False
        new_content = content
        
        for pattern, replacement in replacements:
            if re.search(pattern, new_content):
                new_content = re.sub(pattern, replacement, new_content)
                modified = True
                
        if modified and not self.dry_run:
            file_path.write_text(new_content, encoding="utf-8")
            
        if modified:
            print(f"{'[DRY RUN] ' if self.dry_run else ''}Replaced os.path with pathlib in {file_path}")
            
        return modified
    
    def add_type_hints_to_functions(self, file_path: Path) -> bool:
        """Add basic type hints to function signatures."""
        # This is a complex transformation that would require AST manipulation
        # For now, we'll just identify functions that could benefit from type hints
        
        content = file_path.read_text(encoding="utf-8")
        
        # Find functions without type hints
        function_pattern = r"def (\w+)\([^)]*\):"
        functions_without_hints = []
        
        for match in re.finditer(function_pattern, content):
            func_line = match.group(0)
            if "->" not in func_line and ":" not in func_line.split(")")[0]:
                functions_without_hints.append(match.group(1))
                
        if functions_without_hints:
            print(f"Functions in {file_path} that could benefit from type hints: {functions_without_hints}")
            
        return False  # Don't modify for now, just report
    
    def identify_dataclass_candidates(self, file_path: Path) -> bool:
        """Identify classes that could be converted to dataclasses."""
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
            
            candidates = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Look for simple classes with __init__ that just assigns attributes
                    init_method = None
                    for item in node.body:
                        if (isinstance(item, ast.FunctionDef) and 
                            item.name == "__init__"):
                            init_method = item
                            break
                    
                    if init_method and self._is_simple_init(init_method):
                        candidates.append(node.name)
                        
            if candidates:
                print(f"Classes in {file_path} that could be dataclasses: {candidates}")
                
        except SyntaxError:
            pass  # Skip files with syntax errors
            
        return False
    
    def _is_simple_init(self, init_method: ast.FunctionDef) -> bool:
        """Check if __init__ method is simple enough for dataclass conversion."""
        # Simple heuristic: if all statements are self.attr = param assignments
        for stmt in init_method.body:
            if not isinstance(stmt, ast.Assign):
                return False
            if len(stmt.targets) != 1:
                return False
            target = stmt.targets[0]
            if not (isinstance(target, ast.Attribute) and
                    isinstance(target.value, ast.Name) and
                    target.value.id == "self"):
                return False
        return True
    
    def run_modernization(self) -> None:
        """Run all modernization steps."""
        print(f"Starting modernization of {self.root_path}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY CHANGES'}")
        print("-" * 50)
        
        for py_file in self.find_python_files():
            print(f"\nProcessing {py_file}")
            
            # Add future annotations
            if self.add_future_annotations(py_file):
                self.changes_made += 1
                
            # Standardize typing imports
            if self.standardize_typing_imports(py_file):
                self.changes_made += 1
                
            # Replace os.path with pathlib
            if self.replace_os_path_with_pathlib(py_file):
                self.changes_made += 1
                
            # Identify improvement opportunities
            self.add_type_hints_to_functions(py_file)
            self.identify_dataclass_candidates(py_file)
            
        print(f"\n{'=' * 50}")
        print(f"Modernization complete!")
        print(f"Files modified: {self.changes_made}")
        
        if self.dry_run:
            print("\nThis was a dry run. Use --apply to make actual changes.")
        else:
            print("\nChanges have been applied. Please review and test!")
            
    def run_linting_fixes(self) -> None:
        """Run automated linting fixes."""
        print("Running automated linting fixes...")
        
        commands = [
            ["black", str(self.root_path)],
            ["isort", str(self.root_path)],
            ["ruff", "check", "--fix", str(self.root_path)],
        ]
        
        for cmd in commands:
            if not self.dry_run:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ {cmd[0]} completed successfully")
                    else:
                        print(f"❌ {cmd[0]} failed: {result.stderr}")
                except FileNotFoundError:
                    print(f"⚠️ {cmd[0]} not found, skipping")
            else:
                print(f"[DRY RUN] Would run: {' '.join(cmd)}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Modernize scikit-learn codebase with 2025 best practices"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes to the codebase"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Path to the scikit-learn repository"
    )
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Run linting fixes after modernization"
    )
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        print("Error: Must specify either --dry-run or --apply")
        sys.exit(1)
        
    if args.dry_run and args.apply:
        print("Error: Cannot specify both --dry-run and --apply")
        sys.exit(1)
        
    modernizer = CodeModernizer(args.path, dry_run=args.dry_run)
    modernizer.run_modernization()
    
    if args.lint:
        modernizer.run_linting_fixes()


if __name__ == "__main__":
    main()