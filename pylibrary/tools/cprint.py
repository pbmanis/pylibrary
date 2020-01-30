"""
Routine to perform color printing of text, wrapping standard pyton print function.

"""
# ANSI terminal colors  - just put in as part of the string to get color terminal output
colors = {'red': '\x1b[31m', 'yellow': '\x1b[33m', 'green': '\x1b[32m', 'magenta': '\x1b[35m',
              'blue': '\x1b[34m', 'cyan': '\x1b[36m' , 'white': '\x1b[0m', 'backgray': '\x1b[100m'}

def cprint(c, txt, default='white', **kwds):
    """
    color print: one line
    
    Parameters
    ----------
    c : string
        color, one of [red, yellow, green, magenta, blue, cyan, white, backgray]
    txt : string
        text string to print out
    default: str (default: white)
        default color to return to after printing
    \**kwds:
        Keywords to pass to the print statement (such as end='')
    """

    if c not in list(colors.keys()):
        raise ValueError(f'cprint: color {c:s} is not in valid list of colors')
    if not isinstance(txt, str):
        txt = str(txt)
    print(f"{colors[c]:s}{txt:s}{colors[default]:s}", **kwds)
