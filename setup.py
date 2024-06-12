from setuptools import setup, find_packages

setup(
    name='GSCpy',
    version='1.0.0',
    description='Generalized spectral clustering in Python',
    author='Malik Hacini',
    author_email='mhacini.pro@gmail.com',
    url='https://github.com/Malik-Hacini/GSCpy',  # Update this with your project's URL
    packages=find_packages(where='app'),
    package_dir={'': 'app'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)