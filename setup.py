from setuptools import setup, find_packages

setup(
    name="GACEpy",  # Name of the package
    version="0.1",  # Version of the package
    author="Carlo Musolino",  # Author of the package
    author_email="musolino@itp.uni-frankfurt.de",  # Author's email
    description="GRACEpy provide utilities to ease the submission of GRACE simulations and the analysis of their results.",  # Short description
    long_description=open('README.md').read(),  # Long description from README.md
    long_description_content_type='text/markdown',  # Format of the long description
    url="",  # URL of the project
    packages=find_packages(where="src"),  # Packages to include
    package_dir={"": "src"},  # Root directory of the packages
    classifiers=[  # Additional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Python version requirement
    install_requires=[  # List of dependencies
        "requests",
        "numpy",
        "h5py",
        "argparse",
        "tqdm",
        "vtk",
        "pyparsing"
    ],
    entry_points={  # Command-line tools
        "console_scripts": [
            "archive_source=scripts.archive_source:main",
            "unpack_archive=scripts.unpack_archive:main",  
            "create_descriptor=scripts.create_descriptor:main"
        ],
    },
)
