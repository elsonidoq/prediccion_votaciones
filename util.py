import shutil
from tempfile import mktemp


class safe_write(object):
    def __init__(self, fname):
        self.fname = fname
        self.tmp_fname = mktemp()

    def __enter__(self):
        self.stream = open(self.tmp_fname, 'w')
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            if not self.stream.closed: self.stream.close()
            shutil.move(self.tmp_fname, self.fname)
