from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='polytropon',
    version='0.0.1',
    description='Modular Transformers for multitask learning',
    license="MIT",
    long_description=long_description,
    long_description_context_type='text/markdown',
    url='https://github.com/McGill-NLP/polytropon',
    author='Edoardo Maria Ponti',
    author_email='edoardo-maria.ponti@mila.quebec',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),

    install_requires=[
        "typing>=3.7.4.3",
        "scipy>=1.7.3",
        "torch>=1.10.2",
        "transformers>=4.16.2",
    ],
    python_requires='>=3.9',
)
