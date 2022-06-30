import sys
import app
import tkinter

def main():
    # print(sys.argv)
    # print(type(sys.argv[1]))

    app.App(tkinter.Tk(), "Tkinter and OpenCV", video_source=0, show_video=True)

if __name__ == "__main__":
    main()