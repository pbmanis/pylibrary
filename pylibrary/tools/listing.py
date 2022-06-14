# python
from pathlib import Path
from datetime import datetime

def main():
    thispath = Path(".")

    listing = list(thispath.glob("**/*"))   # recursive glob
    toplevel = sorted([l for l in listing if str(l).find('/') == -1])
    toplevel = sorted(toplevel, key=lambda f: f.stat().st_mtime, reverse=True)
    ldir(toplevel)
    nextlevel = sorted([l for l in listing if str(l).find('/') != -1])
    nextlevel = sorted(nextlevel, key=lambda f: f.stat().st_mtime, reverse=True)
    ldir(nextlevel)
    
def ldir(listing):
    for f in listing:
        if f.is_dir():
            pass
            # print(f"\nDirectory: {str(f):s}")
        else:
            print(f"{str(f):s}\t{datetime.fromtimestamp(f.stat().st_mtime).strftime('%c'):s}\t{f.stat().st_size}")
    # print(list(listing))

# for line in open('ls_dump.txt', 'r'):
#
#     inrec = line.split()
#
#     if inrec == []:
#         continue
#
#     if inrec[0].startswith('total'):
#         continue
#
#     if inrec[0].endswith(':'):
#         folder = inrec[0].replace(':','')
#         continue
#
#     outline = folder + '\t' + '\t'.join(inrec[0:8]) +'\t'+ ' '.join(inrec[8:])
#
#     print( outline )

if __name__ == "__main__":
    main()