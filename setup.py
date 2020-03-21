from setuptools import setup, find_packages

setup(
    name='aem',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tsplib95',
        'numpy',
        'matplotlib',
        'seaborn',
        'tqdm',
        'pandas',
    ],
)
