from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

import sys, re
import setuptools

# (c) Sylvain Corlay, https://github.com/pybind/python_example
def has_flag(compiler, flagname):

  import tempfile

  with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:

    f.write('int main (int argc, char **argv) { return 0; }')

    try:
      compiler.compile([f.name], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
      return False

  return True


# (c) Sylvain Corlay, https://github.com/pybind/python_example
def cpp_flag(compiler):

  if   has_flag(compiler,'-std=c++14'): return '-std=c++14'
  elif has_flag(compiler,'-std=c++11'): return '-std=c++11'
  raise RuntimeError('Unsupported compiler: at least C++11 support is needed')


# (c) Sylvain Corlay, https://github.com/pybind/python_example
class BuildExt(build_ext):

  c_opts = {
    'msvc': ['/EHsc'],
    'unix': [],
  }

  if sys.platform == 'darwin':
    c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

  def build_extensions(self):
    ct = self.compiler.compiler_type
    opts = self.c_opts.get(ct, [])
    if ct == 'unix':
      opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
      opts.append(cpp_flag(self.compiler))
    elif ct == 'msvc':
      opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
    for ext in self.extensions:
      ext.extra_compile_args = opts
    build_ext.build_extensions(self)


ext_modules = [
  Extension(
    'pathfinder',
    sources=['src/pathfinder/PathFinder2.cpp'],
    include_dirs=[
      "libs/pybind11/include/",
    ],
    language='c++'
  ),
]


setup(
  name             = 'pathfinder',
  version           = '0.8',
  description       = 'Implementation of an a*-algorithm in c++',
  url               = 'https://github.com/Tastaturtaste/a_star/tree/pybind/pathfinder',
  ext_modules      = ext_modules,
  packages=find_packages(),
  package_dir={'':'src'},
  install_requires = ['pybind11>=2.2.0'],
  cmdclass         = {'build_ext': BuildExt},
  zip_safe         = False,
)