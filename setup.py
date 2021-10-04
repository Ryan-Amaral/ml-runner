from setuptools import setup

setup(
    name='mlrunner',
    version='0.1',
    packages=['mlrunner'],
    install_requires=["PyTPG@git+https://github.com/Ryan-Amaral/PyTPG.git@v0.9.4.1#egg=PyTPG",
                      "numpy",
                      "gym"], # have to manually install pybullet gym from git repo
    license='GPLv3.0',
    description='Machine learner experiment runner framework.',
    long_description=open('README.md').read(),
    author='Ryan Amaral',
    author_email='ryan_amaral@live.com',
    url='https://github.com/Ryan-Amaral/ml-runner')