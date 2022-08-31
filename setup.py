import codecs
import os

import pkg_resources

from setuptools import find_packages
from setuptools import setup


def run_setup():
    # Read the long description from the README.
    root_path = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(root_path, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    # Read the list of required packages from `requirements.txt`.
    with open(os.path.join(root_path, 'requirements.txt'), 'r') as f:
        required_packages = [
            str(requirement)
            for requirement in pkg_resources.parse_requirements(f)
        ]

    setup(
        name='rl_research',
        version='0.0.1',
        description='RL research code',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Andrii Tytarenko',
        author_email='titarenkoan@gmail.com',
        url='https://github.com/titardrew/rl_research',
        include_package_data=True,
        packages=find_packages(),
        install_requires=required_packages,
        # Supports Python 3 only.
        python_requires='>=3',
        # Add in any packaged data.
        zip_safe=False,
    )


if __name__ == '__main__':
    run_setup()
