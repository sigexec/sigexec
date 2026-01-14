#!/usr/bin/env python3
"""
Script to migrate test files from SignalData/metadata to GraphData/ports.
"""

import re
import sys
from pathlib import Path


def migrate_file(filepath):
    """Migrate a single Python file to use GraphData/ports."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Replace imports
    content = content.replace('from sigexec import SignalData', 'from sigexec import GraphData')
    content = content.replace('from sigexec.core.data import SignalData', 'from sigexec.core.data import GraphData')
    
    # Replace class names
    content = content.replace('SignalData(', 'GraphData(')
    
    # Replace variable names (more conservative)
    content = content.replace('signal_data', 'gdata')
    
    # Replace .metadata references with port access
    # Pattern: gdata.metadata['key'] -> gdata.key
    content = re.sub(r'(\w+)\.metadata\[(["\'])(\w+)\2\]', r'\1.\3', content)
    
    # Pattern: gdata.metadata.get('key', default) -> gdata.get('key', default)  
    content = re.sub(r'(\w+)\.metadata\.get\(', r'\1.get(', content)
    
    # Pattern: 'key' in gdata.metadata -> gdata.has_port('key')
    content = re.sub(r'(["\'])(\w+)\1 in (\w+)\.metadata', r'\3.has_port(\1\2\1)', content)
    
    # Pattern: metadata = gdata.metadata.copy() -> (just remove, we access ports directly)
    content = re.sub(r'\s+metadata = \w+\.metadata\.copy\(\)\s*\n', '', content)
    
    # Pattern: metadata['key'] = value -> gdata.key = value  
    # This is trickier and may need manual review
    
    # Pattern: SignalData(data=..., metadata={...}) needs manual conversion
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Migrated {filepath}")
        return True
    else:
        print(f"- No changes needed for {filepath}")
        return False


def main():
    """Migrate all test files."""
    test_dir = Path('tests')
    
    if not test_dir.exists():
        print(f"Error: {test_dir} not found")
        sys.exit(1)
    
    test_files = list(test_dir.glob('test_*.py'))
    
    print(f"Found {len(test_files)} test files to migrate\n")
    
    migrated = 0
    for test_file in sorted(test_files):
        if migrate_file(test_file):
            migrated += 1
    
    print(f"\n{migrated}/{len(test_files)} files migrated")
    print("\nNote: Some complex patterns may need manual review:")
    print("  - SignalData(data=..., metadata={...}) constructors")
    print("  - metadata['key'] = value assignments")
    print("  - Fixture definitions")


if __name__ == '__main__':
    main()
