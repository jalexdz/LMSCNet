from setuptools import setup, find_packages

setup(
    name='scoutnet',
    version='0.1.0',
    description='Semantic scene completion framework with modular architecture.',
    author='Alex Diaz',
    author_email='j.a.diaz@utexas.edu',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pyyaml',
        'open3d',
        'tqdm',
        'scikit-learn',
        'matplotlib',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            'train = scripts.train:main',
            'validate = scripts.validate:main',
            'predict = scripts.predict:main'
        ]
    },
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.8',
)