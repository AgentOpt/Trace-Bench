#!/usr/bin/env python3
"""
Quick test script to verify kernelbench_simple_test.py can run.
This validates all dependencies without actually running the full pipeline.
"""

import sys
import os

# Add paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kernelbench_root = os.path.join(os.path.dirname(project_root), "KernelBench")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if kernelbench_root not in sys.path:
    sys.path.insert(0, kernelbench_root)


def test_imports():
    """Test all required imports."""
    print("=" * 70)
    print("TESTING IMPORTS")
    print("=" * 70)

    results = []

    # Test datasets
    try:
        from datasets import load_dataset
        print("✓ datasets")
        results.append(True)
    except ImportError as e:
        print(f"✗ datasets: {e}")
        results.append(False)

    # Test litellm
    try:
        import litellm
        print("✓ litellm")
        results.append(True)
    except ImportError as e:
        print(f"✗ litellm: {e}")
        results.append(False)

    # Test modal
    try:
        import modal
        print("✓ modal")
        results.append(True)
    except ImportError as e:
        print(f"✗ modal: {e}")
        results.append(False)

    # Test KernelBench evaluate_with_modal
    try:
        from evaluate_with_modal import app, eval_single_sample_modal, gpu_arch_mapping
        print("✓ evaluate_with_modal (from KernelBench)")
        results.append(True)
    except ImportError as e:
        print(f"✗ evaluate_with_modal: {e}")
        print("   Make sure KernelBench is installed: cd ../../KernelBench && ./install.sh")
        results.append(False)

    return all(results)


def test_api_keys():
    """Test if API keys are configured."""
    print("\n" + "=" * 70)
    print("TESTING API KEYS")
    print("=" * 70)

    results = []

    # Try to load secrets_local
    try:
        from my_processing_agents import secrets_local
        print("✓ secrets_local.py loaded")
    except ImportError:
        print("⚠️  secrets_local.py not found")
        print("   Create: my_processing_agents/secrets_local.py")

    # Check GEMINI_API_KEY
    if os.getenv('GEMINI_API_KEY'):
        key = os.getenv('GEMINI_API_KEY')
        masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
        print(f"✓ GEMINI_API_KEY: {masked}")
        results.append(True)
    else:
        print("✗ GEMINI_API_KEY not set")
        print("   Add to secrets_local.py: os.environ['GEMINI_API_KEY'] = 'your-key'")
        results.append(False)

    return all(results)


def test_modal_auth():
    """Test Modal authentication."""
    print("\n" + "=" * 70)
    print("TESTING MODAL AUTHENTICATION")
    print("=" * 70)

    import subprocess

    try:
        result = subprocess.run(
            ['modal', 'token', 'get'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            print("✓ Modal is authenticated")
            return True
        else:
            print("✗ Modal not authenticated")
            print("   Run: python -m modal setup")
            return False

    except FileNotFoundError:
        print("✗ Modal CLI not found")
        print("   Install: pip install modal")
        return False
    except Exception as e:
        print(f"⚠️  Could not verify Modal auth: {e}")
        return False


def test_dataset_access():
    """Test if we can access the dataset."""
    print("\n" + "=" * 70)
    print("TESTING DATASET ACCESS")
    print("=" * 70)

    try:
        from datasets import load_dataset

        print("Loading dataset (streaming mode)...")
        ds = load_dataset(
            "allenanie/kernelbench_with_prompts",
            split="train",
            streaming=True
        )

        # Get first example
        first_example = next(iter(ds))

        print(f"✓ Dataset loaded")
        print(f"✓ Example keys: {list(first_example.keys())[:5]}")

        # Check expected fields
        if 'backend' in first_example and 'level' in first_example:
            print(f"✓ Has required fields")
            print(f"  Backend: {first_example.get('backend')}")
            print(f"  Level: {first_example.get('level')}")

            # Check if we can filter for level 1 cuda
            if first_example.get('backend') == 'cuda' and first_example.get('level') == 1:
                print("✓ Found level 1 CUDA example")
                return True

        return True

    except Exception as e:
        print(f"✗ Dataset access failed: {e}")
        return False


def test_script_syntax():
    """Test if the script has valid syntax."""
    print("\n" + "=" * 70)
    print("TESTING SCRIPT SYNTAX")
    print("=" * 70)

    script_path = os.path.join(
        project_root,
        'my_processing_agents',
        'kernelbench_simple_test.py'
    )

    try:
        with open(script_path, 'r') as f:
            code = f.read()

        compile(code, script_path, 'exec')
        print(f"✓ Script syntax is valid")
        print(f"  Path: {script_path}")
        return True

    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Could not check syntax: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("KERNELBENCH SIMPLE TEST - SETUP VERIFICATION")
    print("=" * 70)

    results = {
        'imports': test_imports(),
        'api_keys': test_api_keys(),
        'modal_auth': test_modal_auth(),
        'dataset': test_dataset_access(),
        'syntax': test_script_syntax()
    }

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)

    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYou're ready to run:")
        print("  python my_processing_agents/kernelbench_simple_test.py")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nFix the issues above, then run:")
        print("  python my_processing_agents/test_kernelbench_setup.py")

    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
