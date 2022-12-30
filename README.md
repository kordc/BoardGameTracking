# BoardGameTracking

Computer vision project that enables tracking of Cyclades game

Get the video files from shared [folder](https://drive.google.com/drive/folders/1CAKG0U1ZXzRS02MkrOCM3pTlMjBN6GFb?usp=share_link)

This application can be used as CLI app.

Type 
```cmd
    python game_tracker.py --help
```
for the help. Then

```cmd
    python game_tracker.py
``` 
Will run with default setting. If you don't have exactly the same data in the same place this may not work

```cmd
    python game_tracker.py -f=data/cyklady_lvl2_1.mp4
```
To run your custom movie

```cmd
game_tracker.py -f=data/cyklady_lvl1_2.mp4 -e=custom_empty.jpg
```
To additionally give custom empty board reference

```cmd
game_tracker.py -f=data/cyklady_lvl4_5.mp4  -dr=True -dl=True
```
To run this in debugging mode

And you should see something like that:
