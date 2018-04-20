"""
Class object that can be used in place of stdout to write to file as well
as terminal.
"""

# this version 2018 April 6
# Authors: Ben Dive & Alexander Pitchford
# It may be that we have forgotten to credit a source here
# If so, then sorry.

import sys

class DoublePrint(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        try:
            self.log.flush()
        except:
            pass

    def close(self):
        self.flush()
        try:
            self.log.close()
        except:
            pass
