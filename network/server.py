import socket
import threading
import winsound
import os

HEADER = 32
PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, "beep.wav")

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    prev = "good"
    while connected:
        data_length = conn.recv(HEADER).decode(FORMAT) # 1024 is the header size (msg size)
        if data_length: # when connecting to the server, the data_length is empty
            data_length = int(data_length)
            data = conn.recv(data_length).decode(FORMAT)
            print(f"[{addr}] {data}")
            if data == "quit" or data == DISCONNECT_MESSAGE:
                connected = False
                break
            elif "sound" in data and "ALERT" in data and prev == "good":
                winsound.PlaySound(path, winsound.SND_LOOP | winsound.SND_ASYNC)
                prev = "bad"

            elif "good" in data:
                prev = "good"
                winsound.PlaySound(None, winsound.SND_PURGE)

            conn.send(data.encode(FORMAT))
                
    conn.close()
    print(f"[CLOSED CONNECTION] {addr} disconnected.")


def start():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


print("[STARTING] Server is starting...")
print(os.path.dirname(os.path.realpath(__file__)))
start()

