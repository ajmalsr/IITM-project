from moviepy.editor import VideoFileClip,concatenate_videoclips,CompositeVideoClip
c1=VideoFileClip('D:\TRAFFIC\915.mp4')
c2=VideoFileClip('D:\TRAFFIC\925.mp4')
final=concatenate_videoclips([c1,c2])
final.write_videofile("new2.mp4")