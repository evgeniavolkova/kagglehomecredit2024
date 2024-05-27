from setuptools import setup, find_packages

setup(
    name='homecredit',
    version='0.1',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    exclude=[
        "data", 
        "notebooks", 
        "models",
        "scripts", 
        "*.notebooks",
        "data/*",
        "notebooks/*",
        "models/*",
        "scripts/*",
        "archive",
        "archive/*",
        "features",
        "features/*"
    ],
)