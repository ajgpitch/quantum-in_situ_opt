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
        