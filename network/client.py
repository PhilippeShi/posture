import socket
import pickle

HEADER = 32
PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"

class Client:

    def __init__(self, server=SERVER):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((server, PORT))

    def send(self, msg):
        message = msg.encode(FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' ' * (HEADER - len(send_length)) # padding in byte format
        self.client.send(send_length)
        self.client.send(message)
        print(self.client.recv(HEADER).decode(FORMAT))
    
    
    def send_pickle(self, obj):
        '''Send a python object (picture of bad posture)
        so that the server can view it.
        
        Currently not working'''
        
        msg = pickle.dumps(obj)
        msg = bytes(f"{len(msg):<{HEADER}}", "utf-8") + msg
        self.client.send(msg)
    
    def close(self):
        self.send(DISCONNECT_MESSAGE)

if __name__ == "__main__":
    client = Client()
    input()
    client.send("Hello World!")
    # input()
    # client.send_pickle({"name": "John", "age": 30})
    input()
    client.send(DISCONNECT_MESSAGE)