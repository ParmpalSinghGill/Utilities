from __future__ import unicode_literals
import youtube_dl
import os,sys


ydl_opts = {}
os.chdir('.')
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([sys.argv[1]])


#with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#   ydl.download(['https://www.youtube.com/watch?v=JiTz2i4VHFw'])
