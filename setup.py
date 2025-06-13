from setuptools import setup, find_packages

setup(
    name="cmfd_loftr",
    version="0.1.0",
    description="Copy-move forgery detection using LoFTR self-matching.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2",
        "kornia>=0.7.0",
        "opencv-python",
        "scikit-learn",
        "tqdm",
        "pillow",
        "rich"
    ],
    entry_points={
        'console_scripts': [
            'cmfd-loftr=cmfd_loftr.cli:main',
        ],
    },
    python_requires='>=3.8',
) 