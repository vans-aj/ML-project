
from setuptools import setup, find_packages
from typing import List

H = '-e .'

def get_requirements(file_path:str)->List[str]:
    
    requirements=[]
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if H in requirements:
            requirements.remove(H)

    return requirements 

setup(
    name='mlproject',
    version='0.1',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    author='Vansaj Rawat',
    author_email='vnshajrawat951@gmail.com',
)

