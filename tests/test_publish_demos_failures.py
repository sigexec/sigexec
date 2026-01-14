import importlib.util
import pytest
from pathlib import Path


def _load_publish_module():
    spec = importlib.util.spec_from_file_location('publish_demos', Path('examples') / 'publish_demos.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_all_demos_fails_on_broken_demo(tmp_path):
    # Temporarily rename existing demo files so they're not discovered
    examples_dir = Path('examples')
    demo_files = list(examples_dir.glob('*_demo.py'))
    renamed_files = []
    
    for demo_file in demo_files:
        temp_name = demo_file.with_suffix('.py.bak')
        demo_file.rename(temp_name)
        renamed_files.append((demo_file, temp_name))
    
    # Create a broken demo module in examples that will raise when imported
    demo_file = examples_dir / 'broken_demo.py'
    demo_file.write_text('def create_dashboard():\n    raise RuntimeError("broken")\n')

    out_dir = tmp_path / 'out'
    try:
        publish = _load_publish_module().publish_all_demos
        with pytest.raises(SystemExit) as exc:
            publish(str(out_dir), dry_run=True)
        assert exc.value.code == 1
    finally:
        # Clean up
        demo_file.unlink()
        # Restore original demo files
        for original, temp in renamed_files:
            temp.rename(original)
