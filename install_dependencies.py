#!/usr/bin/env python3
"""
Helper script to install dependencies for TMEval.
This script checks for and installs missing dependencies and NLTK data.
"""

import sys
import subprocess
import argparse
import importlib
import importlib.util
import os

def check_package(package_name, required_for=None):
    """Check if a package is installed and print status."""
    is_installed = importlib.util.find_spec(package_name) is not None
    
    if is_installed:
        try:
            # Try to get the version
            package = importlib.import_module(package_name)
            version = getattr(package, '__version__', 'unknown version')
            print(f"✓ {package_name} ({version}) - {'Required for ' + required_for if required_for else 'Core dependency'}")
        except ImportError:
            print(f"✓ {package_name} - {'Required for ' + required_for if required_for else 'Core dependency'}")
    else:
        print(f"✗ {package_name} - {'Required for ' + required_for if required_for else 'Core dependency'}")
    
    return is_installed

def install_package(package_name, upgrade=False):
    """Install a package using pip."""
    print(f"Installing {package_name}...")
    cmd = [sys.executable, "-m", "pip", "install", package_name]
    if upgrade:
        cmd.append("--upgrade")
    
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print(f"✓ Successfully installed {package_name}")
            return True
        else:
            print(f"✗ Failed to install {package_name}")
            return False
    except subprocess.CalledProcessError:
        print(f"✗ Error installing {package_name}")
        return False

def download_nltk_data():
    """Download required NLTK data packages."""
    try:
        import nltk
        
        # Create a directory for NLTK data if it doesn't exist
        nltk_data_dir = os.path.expanduser("~/nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # List of NLTK resources needed by TMEval
        resources = [
            'punkt',           # Sentence tokenizer
            'punkt_tab',
            'wordnet',         # For stemming/lemmatization
            'stopwords'        # Optional, for advanced text processing
        ]
        
        all_success = True
        
        # Check which resources need to be downloaded
        missing_resources = []
        for resource in resources:
            try:
                # Try to find the resource
                resource_path = 'tokenizers/' + resource if resource == 'punkt' else resource
                nltk.data.find(resource_path)
                print(f"✓ NLTK resource '{resource}' is already installed")
            except LookupError:
                missing_resources.append(resource)
        
        # Download missing resources
        if missing_resources:
            print(f"\nDownloading {len(missing_resources)} missing NLTK resources:")
            for resource in missing_resources:
                print(f"Downloading NLTK resource: {resource}")
                try:
                    nltk.download(resource, quiet=False)
                    print(f"✓ Successfully downloaded {resource}")
                except Exception as e:
                    print(f"✗ Error downloading {resource}: {e}")
                    all_success = False
        else:
            print("✓ All required NLTK resources are already installed")
        
        # Verify the resources were downloaded
        print("\nVerifying NLTK resources:")
        all_verified = True
        for resource in resources:
            try:
                resource_path = 'tokenizers/' + resource if resource == 'punkt' else resource
                nltk.data.find(resource_path)
                print(f"✓ Resource {resource} is available")
            except LookupError:
                print(f"✗ Resource {resource} is not available")
                all_verified = False
        
        if all_verified:
            print("\nNLTK data download complete!")
        else:
            print("\nSome NLTK resources could not be verified. You may need to install them manually.")
            print("See README-nltk.md for more information.")
        
        return all_success
        
    except ImportError:
        print("NLTK is not installed. Install it first with 'pip install nltk'")
        return False
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Install dependencies for TMEval")
    parser.add_argument('--all', action='store_true', help='Install all dependencies')
    parser.add_argument('--core', action='store_true', help='Install core dependencies')
    parser.add_argument('--bleu', action='store_true', help='Install BLEU dependencies')
    parser.add_argument('--rouge', action='store_true', help='Install ROUGE dependencies')
    parser.add_argument('--bertscore', action='store_true', help='Install BERTScore dependencies')
    parser.add_argument('--nltk-data', action='store_true', help='Download NLTK data resources')
    parser.add_argument('--upgrade', action='store_true', help='Upgrade existing packages')
    args = parser.parse_args()
    
    # If no specific options, default to checking all packages
    if not (args.all or args.core or args.bleu or args.rouge or args.bertscore or args.nltk_data):
        args.all = True
    
    print("Checking installed packages...")
    
    # Core dependencies
    core_packages = ['pyyaml', 'requests', 'python-dotenv', 'anthropic', 'openai', 'google.generativeai']
    core_missing = []
    
    if args.all or args.core:
        print("\nCore dependencies:")
        for package in core_packages:
            package_name = package.split('.')[0]  # Get the base package name
            if not check_package(package, "Core functionality"):
                core_missing.append(package_name)
    
    # BLEU dependencies
    bleu_packages = ['nltk']
    bleu_missing = []
    
    if args.all or args.bleu:
        print("\nBLEU dependencies:")
        for package in bleu_packages:
            if not check_package(package, "BLEU evaluator"):
                bleu_missing.append(package)
    
    # ROUGE dependencies
    rouge_packages = ['rouge_score', 'nltk']
    rouge_missing = []
    
    if args.all or args.rouge:
        print("\nROUGE dependencies:")
        for package in rouge_packages:
            if not check_package(package, "ROUGE evaluator"):
                rouge_missing.append(package)
    
    # BERTScore dependencies
    bertscore_packages = ['bert_score', 'torch', 'transformers']
    bertscore_missing = []
    
    if args.all or args.bertscore:
        print("\nBERTScore dependencies:")
        for package in bertscore_packages:
            if not check_package(package, "BERTScore evaluator"):
                bertscore_missing.append(package)
    
    # Install missing packages
    missing_packages = []
    if args.all or args.core:
        missing_packages.extend(core_missing)
    if args.all or args.bleu:
        missing_packages.extend(bleu_missing)
    if args.all or args.rouge:
        missing_packages.extend(rouge_missing)
    if args.all or args.bertscore:
        missing_packages.extend(bertscore_missing)
    
    # Remove duplicates
    missing_packages = list(set(missing_packages))
    
    if missing_packages:
        print("\nThe following packages need to be installed:")
        for package in missing_packages:
            print(f"  - {package}")
        
        install = input("\nDo you want to install these packages? [y/N] ")
        if install.lower() == 'y':
            for package in missing_packages:
                install_package(package, args.upgrade)
    else:
        print("\nAll required packages are already installed!")
    
    # Check if NLTK is now installed (either it was already installed or we just installed it)
    nltk_installed = importlib.util.find_spec('nltk') is not None
    
    # Handle NLTK data download
    if nltk_installed and (args.all or args.bleu or args.rouge or args.nltk_data):
        print("\n" + "="*70)
        print("Checking NLTK data resources...")
        
        # Check if NLTK data is already downloaded
        import nltk
        nltk_data_missing = False
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk_data_missing = True
        
        if nltk_data_missing or args.nltk_data:
            if args.nltk_data:
                # If --nltk-data was specified, download without asking
                download_nltk_data()
            else:
                # Otherwise, ask before downloading
                install_nltk = input("NLTK data resources are required. Do you want to download them now? [y/N] ")
                if install_nltk.lower() == 'y':
                    download_nltk_data()
                else:
                    print("NLTK data resources are needed for BLEU and ROUGE evaluators.")
                    print("You can download them later using:")
                    print("  python install_dependencies.py --nltk-data")
        else:
            print("✓ NLTK data resources are already installed")

if __name__ == "__main__":
    print("TMEval Dependency Installer")
    print("===========================")
    main()
    print("\nDone!")
