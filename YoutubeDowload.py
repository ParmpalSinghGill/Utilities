#https://colab.research.google.com/drive/1Z7uJlueMYcN_ZPleQFrd3KqGKbqX5Nwn#scrollTo=urtMiCPbazLg
from __future__ import unicode_literals
import youtube_dl
import os,sys


ydl_opts = {}
os.chdir('.')
# with youtube_dl.YoutubeDL() as ydl:
#     ydl.download([sys.argv[1]])

url="https://www.youtube.com/watch?v=5JohHwNZfQI"

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
  ydl.download([url])

# if not work the then do
# !python3 -m pip install --force-reinstall https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz
# !yt-dlp -f 'bestaudio[ext=m4a]' https://youtu.be/5JohHwNZfQI

