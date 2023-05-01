# Define the video path as a string or a Path object
video_path = "/media/parmpal/Workspace/DataSet/Videos/ਗੁਰੂ ਗੋਬਿੰਦ ਸਿੰਘ ਜੀ ਨੇ ਬੇਦਾਵਾ ਕਿਉਂ ਹਮੇਸ਼ਾ ਆਪਣੇ ਨਾਲ ਰਖਿਆ.mp4"
# text = "ਧੰਨ ਗੁਰੂ ਗੋਬਿੰਦ ਸਿੰਘ ਜੀ"
text = "Hello"
output_path = "output.mp4"

# Import everything needed to edit video clips
from moviepy.editor import *
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

# loading video dsa gfg intro video
clip = VideoFileClip(video_path)

# clipping of the video
# getting video for only starting 10 seconds
clip = clip.subclip(0, 10)

# Reduce the audio volume (volume x 0.8)
clip = clip.volumex(0.8)

# Generate a text clip
txt_clip = TextClip(text, fontsize=75, color='black')

# setting position of text in the center and duration will be 10 seconds
txt_clip = txt_clip.set_pos('center').set_duration(10)

# Overlay the text clip on the first video clip
video = CompositeVideoClip([clip, txt_clip])

# # showing video
# video.ipython_display(width=280)

video.write_videofile(output_path)
