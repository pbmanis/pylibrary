"""
Routine to perform color printing of text, wrapping standard pyton print function.

"""
# ANSI terminal colors  - just put in as part of the string to get color terminal output
colors = {'red': '\x1b[31m', 'r': '\x1b[31m', 
          'orange': '\x1b[48:5:208:0m', 'o': '\x1b[48:5:208:0m',
          'yellow': '\x1b[33m', 'y': '\x1b[33m',
          'green': '\x1b[32m', 'g': '\x1b[32m', 
          'magenta': '\x1b[35m', 'm': '\x1b[35m',
          'blue': '\x1b[34m', 'b': '\x1b[34m',
          'cyan': '\x1b[36m' , 'c': '\x1b[36m' ,
          'white': '\x1b[0m', 'w': '\x1b[0m',
          'backgray': '\x1b[100m'}

def cprint(c, txt, default='white', textonly:bool=False, **kwds):
    r"""
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
    text = f"{colors[c]:s}{txt:s}{colors[default]:s}"
    if not textonly:
        print(text, **kwds)
        return None
    else:
        return text
    
if __name__ == "__main__":
    # do some tests
    print('testing cprint')
    for c in colors.keys():
        cprint(c, c)
    
    print('testing cprint with textonly')
    txt = " "
    for c in colors.keys():
        txt= "\n".join([txt, cprint(c, c, textonly=True, end="")])
    print(txt)
