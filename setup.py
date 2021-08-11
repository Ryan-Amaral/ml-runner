from setuptools import setup

setup(
    name='mlrunner',
    version='0.1',
    packages=['mlrunner'],
    install_requires=["PyTPG@git+https://github.com/Ryan-Amaral/PyTPG.git@v0.9.4.2#egg=PyTPG",
                      "numpy",
                      "gym",
                      "box2d-py"], # enter requirements here
    license='GPLv3.0',
    description='Machine learner experiment runner framework.',
    long_description=open('README.md').read(),
    author='Ryan Amaral',
    author_email='ryan_amaral@live.com',
    url='https://github.com/Ryan-Amaral/ml-runner')