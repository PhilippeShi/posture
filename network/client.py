import socket

HEADER = 64
PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"

class Client:

    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(ADDR)

    def send(self, msg):
        message = msg.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' ' * (HEADER - len(send_length)) # padding in byte format
        self.client.send(send_length)
        self.client.send(message)
        print(self.client.recv(2048).decode(FORMAT))
    
    def close(self):
        self.send(DISCONNECT_MESSAGE)

if __name__ == "__main__":
    client = Client()
    client.send("Hello World!")
    input()
    client.send("MSG 2")
    input()
    client.send(DISCONNECT_MESSAGE)