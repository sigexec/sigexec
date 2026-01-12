"""
Publish Demos

This script automatically discovers and publishes all demo dashboards.
Run this to generate the complete docs/ site with all demos,
or run individual demo files to publish just that demo to staticdash/.

To add a new demo:
1. Create a new *_demo.py file in examples/
2. Implement a create_dashboard() function that returns a sd.Dashboard
3. Run this script - it will automatically pick up the new demo
"""

import sys
import os
import importlib.util
from pathlib import Path

try:
    import staticdash as sd
    STATICDASH_AVAILABLE = True
except ImportError:
    STATICDASH_AVAILABLE = False
    print("Error: staticdash not installed. Install with: pip install staticdash")
    sys.exit(1)


def discover_demo_modules():
    """
    Discover all *_demo.py files in the examples directory.
    
    Returns:
        List of (module_name, module_object, slug) tuples
    """
    examples_dir = Path(__file__).parent
    demo_files = sorted(examples_dir.glob('*_demo.py'))
    
    modules = []
    for demo_file in demo_files:
        module_name = demo_file.stem
        slug = module_name.replace('_', '-')
        
        # Dynamically import the module
        spec = importlib.util.spec_from_file_location(module_name, demo_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                # Verify it has create_dashboard function
                if hasattr(module, 'create_dashboard'):
                    modules.append((module_name, module, slug))
                else:
                    print(f"Warning: {demo_file.name} missing create_dashboard() function, skipping")
            except Exception as e:
                print(f"Error loading {demo_file.name}: {e}")
    
    return modules


def publish_all_demos(output_dir='docs'):
    """
    Discover and run all demo files, publishing them to a single directory.
    
    Args:
        output_dir: Directory to publish to (default: 'docs' for GitHub Pages)
    """
    print("\n" + "="*70)
    print("Publishing All sigexec Demos")
    print("="*70 + "\n")
    
    # Discover demo modules
    demo_modules = discover_demo_modules()
    
    if not demo_modules:
        print("No demo modules found!")
        return
    
    print(f"Found {len(demo_modules)} demo(s):\n")
    for module_name, _, slug in demo_modules:
        print(f"  - {module_name} → {slug}/")
    print()
    
    # Create directory to hold multiple dashboards
    directory = sd.Directory(
        title='sigexec demos',
        page_width=1000
    )
    
    # Create and add each dashboard
    slugs = []
    for module_name, module, slug in demo_modules:
        print(f"Creating {module_name}...")
        try:
            dashboard = module.create_dashboard()
            directory.add_dashboard(dashboard, slug=slug)
            slugs.append((slug, dashboard.title))
            print(f"  ✓ Success")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Publish everything to output directory
    print(f"\nPublishing dashboards to {output_dir}/...")
    directory.publish(output_dir)
    
    print(f"\n{'='*70}")
    print("✓ Dashboards created successfully!")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  {output_dir}/index.html - Landing page with all dashboards")
    for slug, title in slugs:
        print(f"  {output_dir}/{slug}/ - {title}")
    print(f"\nTo view: Open {output_dir}/index.html in a web browser")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Default: publish to docs/ for GitHub Pages
    output_dir = 'docs'
    
    # Allow custom output directory from command line
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    publish_all_demos(output_dir)
