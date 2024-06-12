from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='GSCpy',
    version='0.1.1',
    description='Generalized Spectral Clustering in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Malik Hacini',
    author_email='mhacini.pro@gmail.com',
    url='https://github.com/Malik-Hacini/GSCpy',  # Update this with your project's URL
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=['numpy','matplotlib','scipy','sklearn']
)