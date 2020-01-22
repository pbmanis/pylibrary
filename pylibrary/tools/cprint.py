
# ANSI terminal colors  - just put in as part of the string to get color terminal output
colors = {'red': '\x1b[31m', 'yellow': '\x1b[33m', 'green': '\x1b[32m', 'magenta': '\x1b[35m',
              'blue': '\x1b[34m', 'cyan': '\x1b[36m' , 'white': '\x1b[0m', 'backgray': '\x1b[100m'}

def cprint(c, txt):
    """
    color print: one line
    """
    if not isinstance(txt, str):
        txt = str(txt)
    print(f"{colors[c]:s}{txt:s}{colors['white']:s}")
